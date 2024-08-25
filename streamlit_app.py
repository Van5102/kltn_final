import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt
from streamlit.logger import get_logger
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
import chardet

LOGGER = get_logger(__name__)
df = None
df_cluster = None

def read_csv_with_file_uploader():
    uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        encoding = chardet.detect(bytes_data)['encoding']
        stringio = io.StringIO(bytes_data.decode(encoding))
        return pd.read_csv(stringio)
    return None

def input_file():
    global df, df_cluster
    df = read_csv_with_file_uploader()
    if df is not None:
        df = df.dropna().head(2000)
        st.dataframe(df)
        st.write('Loại bỏ các giá trị Null:')
        st.write(df.describe())
        
        selected_columns = st.multiselect('Lựa chọn dữ liệu phân cụm:', df.columns.to_list())
        if len(selected_columns) > 3:
            st.error('Chỉ lựa chọn tối đa 3 cột dữ liệu để phân cụm.')
            return None
        
        if selected_columns:
            df_cluster = df[selected_columns]
            st.dataframe(df_cluster)
            Elbow(df_cluster)
            return df_cluster
    return None

def export_clustered_data():
    if df is not None and 'Cluster' in df.columns:
        data = df.sort_values('Cluster')
        output_filename = 'clustered_data.csv'
        data_csv = data.to_csv(index=False)
        st.download_button(
            label='TẢI VỀ KẾT QUẢ PHÂN CỤM',
            data=data_csv,
            file_name=output_filename
        )
        st.dataframe(data)
    else:
        st.write('No data to export.')

def runKmean(df_cluster, n):
    st.title('Biểu đồ phân cụm')
    if df_cluster is not None:
        kmeans = KMeans(n_clusters=n, init='k-means++', max_iter=300, n_init=10)
        clusters = kmeans.fit_predict(df_cluster)
        df_cluster['Cluster'] = clusters
        centroids = kmeans.cluster_centers_
        cluster_counts = df_cluster['Cluster'].value_counts()

        if df_cluster.shape[1] > 2:
            # Create a 3D scatter plot of the clusters
            fig = go.Figure()
            colors = px.colors.qualitative.Plotly
            
            for i in range(n):
                cluster_df = df_cluster[df_cluster['Cluster'] == i]
                fig.add_trace(go.Scatter3d(
                    x=cluster_df.iloc[:, 0],
                    y=cluster_df.iloc[:, 1],
                    z=cluster_df.iloc[:, 2],
                    mode='markers',
                    marker=dict(size=3, color=colors[i % len(colors)]),
                    name=f'Cluster {i}'
                ))
                for _, row in cluster_df.iterrows():
                    fig.add_trace(go.Scatter3d(
                        x=[centroids[i][0], row.iloc[0]],
                        y=[centroids[i][1], row.iloc[1]],
                        z=[centroids[i][2], row.iloc[2]],
                        mode='lines',
                        line=dict(color=colors[i % len(colors)], width=2),
                        showlegend=False
                    ))
            
            fig.add_trace(go.Scatter3d(
                x=centroids[:, 0],
                y=centroids[:, 1],
                z=centroids[:, 2],
                mode='markers',
                marker=dict(size=6, color='black'),
                name='Centroids'
            ))
            
            fig.update_layout(
                scene=dict(
                    xaxis_title=df_cluster.columns[0],
                    yaxis_title=df_cluster.columns[1],
                    zaxis_title=df_cluster.columns[2]
                ),
                legend=dict(title='', itemsizing='constant')
            )
            
            st.plotly_chart(fig)
        else:
            # Create a 2D scatter plot for 2D data
            plt.figure(figsize=(10, 6))
            scatter = plt.scatter(
                df_cluster.iloc[:, 0],
                df_cluster.iloc[:, 1],
                c=clusters,
                cmap='viridis',
                marker='o'
            )
            centers = kmeans.cluster_centers_
            plt.scatter(
                centers[:, 0],
                centers[:, 1],
                c='red',
                s=200,
                alpha=0.75,
                marker='x'
            )
            plt.xlabel(df_cluster.columns[0])
            plt.ylabel(df_cluster.columns[1])
            plt.legend(*scatter.legend_elements(), title='Clusters')
            st.pyplot()
        
        st.write('Số lượng điểm dữ liệu trong mỗi cụm:', cluster_counts)
    return df_cluster

def find_optimal_eps_min_samples(df_cluster):
    nearest_neighbors = NearestNeighbors(n_neighbors=2)
    distances, _ = nearest_neighbors.fit(df_cluster).kneighbors(df_cluster)
    distances = np.sort(distances[:, 1])
    kneedle = KneeLocator(range(1, len(distances) + 1), distances, curve='convex', direction='increasing')
    eps = distances[kneedle.elbow]
    min_samples = 2 * df_cluster.shape[1]
    st.write('Giá trị eps tối ưu:', eps)
    st.write('Giá trị min_samples tối ưu:', min_samples)
    return eps, min_samples

def runDbScan(df_cluster):
    radio_button = st.radio('Lựa chọn giá trị eps và min_samples', ['Tối ưu', 'Tự nhập'])
    if radio_button == 'Tối ưu':
        eps, min_samples = find_optimal_eps_min_samples(df_cluster)
    else:
        eps = st.slider('Chọn giá trị eps', min_value=0.1, max_value=100.0, value=0.1, step=0.1)
        min_samples = st.slider('Chọn giá trị min_samples', min_value=1, max_value=200, value=5, step=1)
    
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(df_cluster)
    df_cluster['Cluster'] = clusters
    st.set_option('deprecation.showPyplotGlobalUse', False)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(
        df_cluster.iloc[:, 0],
        df_cluster.iloc[:, 1],
        c=clusters,
        cmap='viridis',
        marker='o'
    )
    for i, txt in enumerate(clusters):
        if txt != -1:
            plt.annotate(f'{txt}', (df_cluster.iloc[i, 0], df_cluster.iloc[i, 1]))
    
    plt.title('DBSCAN Clustering')
    plt.xlabel(df_cluster.columns[0])
    plt.ylabel(df_cluster.columns[1])
    st.pyplot()
    
    cluster_counts = df_cluster['Cluster'].value_counts()
    cluster_counts = cluster_counts[cluster_counts.index != -1]
    n_noise = list(clusters).count(-1)
    st.write('Số lượng cụm:', len(cluster_counts))
    st.write('Số lượng điểm nhiễu:', n_noise)
    st.write('Số lượng điểm dữ liệu trong mỗi cụm:')
    st.dataframe(cluster_counts)
    return df_cluster

def Elbow(df_cluster):
    st.title('Chọn số cụm tối ưu bằng phương pháp Elbow')
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(df_cluster)
        wcss.append(kmeans.inertia_)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, 11), wcss, marker='o', linestyle='--')
    ax.set_title('Elbow Method')
    ax.set_xlabel('Number of clusters')
    ax.set_ylabel('WCSS')
    
    st.pyplot(fig)

def run():
    global df, df_cluster
    st.set_page_config(page_title='Demo Sản Phẩm', page_icon='💻')

    with st.sidebar:
        st.title('Menu')
        radio_selection = st.radio('Lựa chọn thuật toán', ['K-MEANS', 'DBSCAN'])
    
    st.title('Các thuật toán học máy trong khai thác dữ liệu lớn và ứng dụng phân đoạn khách hàng')
    if radio_selection == 'K-MEANS':
        st.markdown("<h1 style='text-align: center;'>KMEAN CLUSTERING</h1>", unsafe_allow_html=True)
    else:
        st.markdown("<h1 style='text-align: center;'>DBSCAN CLUSTERING</h1>", unsafe_allow_html=True)
    
    df_cluster = input_file()
    if df_cluster is not None:
        if radio_selection == 'K-MEANS':
            n = st.number_input('Nhập số cụm', min_value=2, key=int)
            df_cluster = runKmean(df_cluster, n)
        else:
            df_cluster = runDbScan(df_cluster)
        if df_cluster is not None and 'Cluster' in df_cluster.columns:
            df['Cluster'] = df_cluster['Cluster']
            export_clustered_data()

if __name__ == '__main__':
    run()
