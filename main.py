from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import uvicorn
import os

# Armazenar o HTML diretamente como uma string para evitar problemas com sistema de arquivos no Vercel
INDEX_HTML = """
<!DOCTYPE html>
<html lang="pt-br">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Análise de Clientes e Campanhas</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f5f7fa;
            color: #333;
        }
        .container {
            width: 90%;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        header {
            background-color: #4a6fa5;
            color: white;
            padding: 20px 0;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        h1 {
            margin: 0;
            font-size: 2.5em;
        }
        .upload-section {
            background-color: white;
            border-radius: 8px;
            padding: 30px;
            margin: 30px 0;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
        }
        .file-input {
            border: 2px dashed #ddd;
            padding: 20px;
            border-radius: 5px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
        }
        .file-input:hover {
            border-color: #4a6fa5;
            background-color: #f0f5ff;
        }
        input[type="file"] {
            display: none;
        }
        .btn {
            background-color: #4a6fa5;
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            transition: background-color 0.3s;
        }
        .btn:hover {
            background-color: #3a5a8a;
        }
        .results-section {
            background-color: white;
            border-radius: 8px;
            padding: 30px;
            margin: 30px 0;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
        }
        .chart-container {
            margin: 30px 0;
            text-align: center;
        }
        .chart {
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #f2f6fc;
            font-weight: 600;
        }
        tr:hover {
            background-color: #f5f7fa;
        }
        .cluster-info {
            background-color: #f2f6fc;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
            border-left: 5px solid #4a6fa5;
        }
        .recommendations {
            background-color: #edf7ed;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
            border-left: 5px solid #4caf50;
        }
        .section-title {
            border-bottom: 2px solid #e0e6ed;
            padding-bottom: 10px;
            margin-top: 40px;
            color: #4a6fa5;
        }
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }
        .spinner {
            border: 5px solid #f3f3f3;
            border-top: 5px solid #4a6fa5;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        #file-name-transacoes, #file-name-campanhas {
            margin-top: 10px;
            font-size: 14px;
            color: #666;
        }
        .hidden {
            display: none;
        }
    </style>
</head>
<body>
    <header>
        <div class="container">
            <h1>Análise de Clientes e Campanhas</h1>
            <p>Carregue seus arquivos CSV para análise detalhada</p>
        </div>
    </header>

    <div class="container">
        <section class="upload-section">
            <h2>Upload de Dados</h2>
            <form id="upload-form" action="/analyze" method="post" enctype="multipart/form-data">
                <div class="form-group">
                    <label for="file-transacoes">Arquivo de Transações (.csv)</label>
                    <div class="file-input" onclick="document.getElementById('file-transacoes').click()">
                        <p>Clique para selecionar arquivo de transações</p>
                        <div id="file-name-transacoes"></div>
                    </div>
                    <input type="file" id="file-transacoes" name="file_transacoes" accept=".csv" onchange="updateFileName(this, 'file-name-transacoes')">
                </div>
                
                <div class="form-group">
                    <label for="file-campanhas">Arquivo de Campanhas (.csv)</label>
                    <div class="file-input" onclick="document.getElementById('file-campanhas').click()">
                        <p>Clique para selecionar arquivo de campanhas</p>
                        <div id="file-name-campanhas"></div>
                    </div>
                    <input type="file" id="file-campanhas" name="file_campanhas" accept=".csv" onchange="updateFileName(this, 'file-name-campanhas')">
                </div>
                
                <button type="submit" class="btn" onclick="showLoading()">Analisar Dados</button>
            </form>
            
            <div id="loading" class="loading">
                <div class="spinner"></div>
                <p>Analisando dados, por favor aguarde...</p>
            </div>
        </section>

        <section id="results-section" class="results-section hidden">
            <h2 class="section-title">Resultados da Análise</h2>
            
            <h3>Clusterização de Clientes</h3>
            <div class="chart-container">
                <img id="cluster-chart" class="chart" src="" alt="Gráfico de Clusters">
            </div>
            
            <div id="cluster-info">
                <!-- Será preenchido dinamicamente -->
            </div>
            
            <h3 class="section-title">Análise de Campanhas</h3>
            <div class="chart-container">
                <img id="campaign-chart-1" class="chart" src="" alt="Gasto Médio por Cliente por Campanha">
            </div>
            <div class="chart-container">
                <img id="campaign-chart-2" class="chart" src="" alt="ROI Estimado por Campanha">
            </div>
            
            <h3 class="section-title">Impacto das Campanhas</h3>
            <div class="chart-container">
                <img id="regression-chart" class="chart" src="" alt="Impacto das Campanhas no Total Gasto">
            </div>
            
            <div id="regression-info">
                <!-- Coeficientes da regressão aqui -->
            </div>
            
            <h3 class="section-title">Análise de CLV</h3>
            <div class="chart-container">
                <img id="clv-chart" class="chart" src="" alt="Distribuição do CLV por Segmento">
            </div>
            
            <h3 class="section-title">Clientes de Alto Valor</h3>
            <div id="high-value-clients">
                <!-- Tabela de clientes de alto valor -->
            </div>
            
            <h3 class="section-title">Recomendações de Marketing</h3>
            <div class="recommendations" id="recommendations">
                <!-- Recomendações aqui -->
            </div>
        </section>
    </div>

    <script>
        function updateFileName(input, elementId) {
            const fileNameDiv = document.getElementById(elementId);
            if (input.files.length > 0) {
                fileNameDiv.textContent = `Arquivo selecionado: ${input.files[0].name}`;
            } else {
                fileNameDiv.textContent = '';
            }
        }
        
        function showLoading() {
            if (validateForm()) {
                document.getElementById('loading').style.display = 'block';
                return true;
            }
            return false;
        }
        
        function validateForm() {
            const transacoesFile = document.getElementById('file-transacoes').files[0];
            const campanhasFile = document.getElementById('file-campanhas').files[0];
            
            if (!transacoesFile || !campanhasFile) {
                alert('Por favor, selecione ambos os arquivos CSV antes de analisar.');
                return false;
            }
            return true;
        }
    </script>
</body>
</html>
    """)

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    return HTMLResponse(content=INDEX_HTML)

@app.post("/analyze", response_class=HTMLResponse)
async def analyze_data(
    request: Request,
    file_transacoes: UploadFile = File(...),
    file_campanhas: UploadFile = File(...)
):
    # Ler arquivos CSV
    transacoes_content = await file_transacoes.read()
    campanhas_content = await file_campanhas.read()
    
    transacoes = pd.read_csv(io.BytesIO(transacoes_content))
    campanhas = pd.read_csv(io.BytesIO(campanhas_content))
    
    # Resultados da análise
    results = {}
    
    # 1. Clusterização de Clientes
    clientes = transacoes.groupby('cliente_id').agg({
        'frequencia_compras': 'max',
        'total_gasto': 'max',
        'ultima_compra': 'max'
    }).reset_index()
    
    scaler = StandardScaler()
    clientes_scaled = scaler.fit_transform(clientes[['frequencia_compras', 'total_gasto', 'ultima_compra']])
    
    kmeans = KMeans(n_clusters=3, random_state=42)
    clientes['cluster'] = kmeans.fit_predict(clientes_scaled)
    
    pca = PCA(n_components=2)
    clientes[['pca1', 'pca2']] = pca.fit_transform(clientes_scaled)
    
    # Gráfico de clusters
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=clientes, x='pca1', y='pca2', hue='cluster', palette='Set2')
    plt.title('Clusters de Clientes')
    
    cluster_img = io.BytesIO()
    plt.savefig(cluster_img, format='png', bbox_inches='tight')
    cluster_img.seek(0)
    cluster_img_b64 = base64.b64encode(cluster_img.read()).decode()
    results['cluster_chart'] = f"data:image/png;base64,{cluster_img_b64}"
    plt.close()
    
    # Diagnóstico por cluster
    cluster_diagnostico = clientes.groupby('cluster')[['frequencia_compras', 'total_gasto', 'ultima_compra']].mean().round(2)
    results['cluster_info'] = []
    
    for idx, row in cluster_diagnostico.iterrows():
        cluster_type = ""
        if row['frequencia_compras'] > 12 and row['total_gasto'] > 5000:
            cluster_type = "Cliente fiel e de alto valor"
        elif row['ultima_compra'] > 250:
            cluster_type = "Cliente inativo"
        else:
            cluster_type = "Cliente de valor médio e recorrência moderada"
        
        results['cluster_info'].append({
            'cluster': idx,
            'type': cluster_type,
            'frequencia': row['frequencia_compras'],
            'gasto': row['total_gasto'],
            'ultima_compra': row['ultima_compra']
        })
    
    # Merge dos clusters nos dados de transações
    transacoes = pd.merge(transacoes, clientes[['cliente_id', 'cluster']], on='cliente_id', how='left')
    
    # 2. Análise de Preferência por Campanhas
    preferencia_campanhas = transacoes.groupby('campanha').agg({
        'cliente_id': 'nunique',
        'valor_compra': 'sum',
        'frequencia_compras': 'sum',
        'total_gasto': 'sum'
    }).reset_index()
    
    preferencia_campanhas = pd.merge(preferencia_campanhas, 
                                     campanhas, 
                                     left_on='campanha', 
                                     right_on='nome_campanha', 
                                     how='left')
    
    preferencia_campanhas['gasto_medio_por_cliente'] = preferencia_campanhas['total_gasto'] / preferencia_campanhas['cliente_id']
    preferencia_campanhas['roi_estimado'] = preferencia_campanhas['total_gasto'] / preferencia_campanhas['custo_campanha']
    
    # Gráfico de gasto médio por cliente por campanha
    plt.figure(figsize=(12, 6))
    sns.barplot(data=preferencia_campanhas, x='campanha', y='gasto_medio_por_cliente')
    plt.title('Gasto Médio por Cliente por Campanha')
    plt.xticks(rotation=45)
    
    campaign_chart1_img = io.BytesIO()
    plt.savefig(campaign_chart1_img, format='png', bbox_inches='tight')
    campaign_chart1_img.seek(0)
    campaign_chart1_img_b64 = base64.b64encode(campaign_chart1_img.read()).decode()
    results['campaign_chart1'] = f"data:image/png;base64,{campaign_chart1_img_b64}"
    plt.close()
    
    # Gráfico de ROI estimado por campanha
    plt.figure(figsize=(12, 6))
    sns.barplot(data=preferencia_campanhas, x='campanha', y='roi_estimado')
    plt.title('ROI Estimado por Campanha')
    plt.xticks(rotation=45)
    
    campaign_chart2_img = io.BytesIO()
    plt.savefig(campaign_chart2_img, format='png', bbox_inches='tight')
    campaign_chart2_img.seek(0)
    campaign_chart2_img_b64 = base64.b64encode(campaign_chart2_img.read()).decode()
    results['campaign_chart2'] = f"data:image/png;base64,{campaign_chart2_img_b64}"
    plt.close()
    
    # 3. Regressão Linear para avaliar impacto das campanhas
    transacoes_reg = transacoes.merge(campanhas, left_on='campanha', right_on='nome_campanha', how='left')
    features = ['custo_campanha', 'alcance', 'conversao']
    X = transacoes_reg[features]
    y = transacoes_reg['total_gasto']
    
    reg_model = LinearRegression()
    reg_model.fit(X, y)
    coef_df = pd.DataFrame({'Variavel': features, 'Coeficiente': reg_model.coef_})
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=coef_df, x='Variavel', y='Coeficiente')
    plt.title('Impacto das Campanhas no Total Gasto')
    plt.ylabel('Coeficiente da Regressão')
    
    regression_img = io.BytesIO()
    plt.savefig(regression_img, format='png', bbox_inches='tight')
    regression_img.seek(0)
    regression_img_b64 = base64.b64encode(regression_img.read()).decode()
    results['regression_chart'] = f"data:image/png;base64,{regression_img_b64}"
    plt.close()
    
    results['regression_info'] = coef_df.sort_values(by='Coeficiente', ascending=False).to_dict('records')
    
    # 4. Análise de CLV (Customer Lifetime Value)
    clientes_clv = transacoes[['cliente_id', 'total_gasto']].drop_duplicates()
    clientes_clv.rename(columns={'total_gasto': 'clv'}, inplace=True)
    
    thresh_clv = clientes_clv['clv'].quantile(0.75)
    clientes_clv['segmento_valor'] = clientes_clv['clv'].apply(lambda x: 'Alto Valor' if x >= thresh_clv else 'Demais')
    
    plt.figure(figsize=(12, 6))
    sns.histplot(data=clientes_clv, x='clv', hue='segmento_valor', bins=30, kde=True, palette='Set2')
    plt.title('Distribuição do CLV por Segmento')
    plt.xlabel('Customer Lifetime Value')
    plt.ylabel('Frequência')
    
    clv_img = io.BytesIO()
    plt.savefig(clv_img, format='png', bbox_inches='tight')
    clv_img.seek(0)
    clv_img_b64 = base64.b64encode(clv_img.read()).decode()
    results['clv_chart'] = f"data:image/png;base64,{clv_img_b64}"
    plt.close()
    
    # 5. Clientes de Alto Valor (gasto total >= 60000)
    clientes_alto_gasto = transacoes[transacoes['total_gasto'] >= 60000]
    results['high_value_clients'] = clientes_alto_gasto.to_dict('records')
    
    # 6. Recomendações de Marketing
    results['recommendations'] = [
        "Priorizar campanhas com ROI elevado, como aquelas que entregaram maior retorno por real investido.",
        "Reavaliar ou reformular campanhas com ROI baixo, focando em novos formatos ou incentivos como brindes, frete grátis, etc.",
        "Investir mais em campanhas com alto gasto médio por cliente, pois indicam maior valor percebido.",
        "Oferecer experiências exclusivas e personalizadas para clientes de alto valor, como eventos VIP ou convites para lançamentos de produtos.",
        "Criar um programa de fidelidade premium com recompensas e benefícios exclusivos.",
        "Desenvolver um sistema de recomendação de produtos baseado no histórico de compra de cada cliente."
    ]
    
    # Construir o HTML de resposta dinamicamente
    result_html = INDEX_HTML.replace('id="results-section" class="results-section hidden"', 'id="results-section" class="results-section"')
    
    # Injetar imagens nos pontos corretos
    result_html = result_html.replace('id="cluster-chart" class="chart" src=""', 
                                     f'id="cluster-chart" class="chart" src="{results["cluster_chart"]}"')
    result_html = result_html.replace('id="campaign-chart-1" class="chart" src=""', 
                                     f'id="campaign-chart-1" class="chart" src="{results["campaign_chart1"]}"')
    result_html = result_html.replace('id="campaign-chart-2" class="chart" src=""', 
                                     f'id="campaign-chart-2" class="chart" src="{results["campaign_chart2"]}"')
    result_html = result_html.replace('id="regression-chart" class="chart" src=""', 
                                     f'id="regression-chart" class="chart" src="{results["regression_chart"]}"')
    result_html = result_html.replace('id="clv-chart" class="chart" src=""', 
                                     f'id="clv-chart" class="chart" src="{results["clv_chart"]}"')
    
    # Adicionar informações de clusters
    cluster_info_html = ""
    for info in results['cluster_info']:
        cluster_info_html += f"""
        <div class="cluster-info">
            <h4>Cluster {info['cluster']}: {info['type']}</h4>
            <p>Frequência média de compras: {info['frequencia']}</p>
            <p>Gasto total médio: R$ {info['gasto']:.2f}</p>
            <p>Dias desde última compra (média): {info['ultima_compra']}</p>
        </div>
        """
    result_html = result_html.replace('id="cluster-info">', f'id="cluster-info">{cluster_info_html}')
    
    # Adicionar informações de regressão
    regression_info_html = "<table><tr><th>Variável</th><th>Coeficiente</th></tr>"
    for info in results['regression_info']:
        regression_info_html += f"<tr><td>{info['Variavel']}</td><td>{info['Coeficiente']:.4f}</td></tr>"
    regression_info_html += "</table>"
    result_html = result_html.replace('id="regression-info">', f'id="regression-info">{regression_info_html}')
    
    # Adicionar clientes de alto valor
    if results['high_value_clients']:
        high_value_html = "<table><tr><th>Cliente ID</th><th>Total Gasto</th><th>Frequência</th><th>Última Compra</th></tr>"
        for client in results['high_value_clients'][:10]:  # Limitando a 10 para não sobrecarregar
            high_value_html += f"<tr><td>{client['cliente_id']}</td><td>R$ {client['total_gasto']:.2f}</td>"
            high_value_html += f"<td>{client['frequencia_compras']}</td><td>{client['ultima_compra']} dias</td></tr>"
        high_value_html += "</table>"
        result_html = result_html.replace('id="high-value-clients">', f'id="high-value-clients">{high_value_html}')
    
    # Adicionar recomendações
    recommendations_html = "<ul>"
    for rec in results['recommendations']:
        recommendations_html += f"<li>{rec}</li>"
    recommendations_html += "</ul>"
    result_html = result_html.replace('id="recommendations">', f'id="recommendations">{recommendations_html}')
    
    return HTMLResponse(content=result_html)

@app.get("/health")
async def health_check():
    return {"status": "ok"}

# Código para iniciar localmente (não usado no Vercel)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
