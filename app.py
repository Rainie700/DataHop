"""
投資組合分析系統 - 主程式
Portfolio Analysis System - Main Application
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from portfolio_analyzer import (
    get_returns_matrix,
    optimize_portfolio,
    monte_carlo_simulation,
    stress_test,
    backtest,
    project_returns,
)

# Page config
st.set_page_config(
    page_title="DataHop",
    page_icon="images/logo.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .risk-conservative { color: #2ecc71; }
    .risk-moderate { color: #f39c12; }
    .risk-aggressive { color: #e74c3c; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
</style>
""", unsafe_allow_html=True)
#st.image("images/logo2.png", width=200)
st.markdown('<p class="main-header">投資組合分析系統</p>', unsafe_allow_html=True)
st.caption("輸入本金與股票代碼，系統將計算最佳配置、風險分析、壓力測試與蒙地卡羅模擬")

# Sidebar inputs
with st.sidebar:
    
    # 新增兩行
    from PIL import Image
    st.image("images/logo2.png", width=150)
    
    st.header("參數設定")
    principal = st.number_input(
        "本金 (NTD/USD)",
        min_value=10000,
        value=1000000,
        step=10000,
        format="%d"
    )

    ticker_input = st.text_area(
        "股票代碼 (每行一檔，2-10檔)",
        value="2330.TW\n2454.TW\n2317.TW",
        height=120,
        help="範例：台股 2330.TW、美股 AAPL。Yahoo Finance 格式"
    )

    tickers = [t.strip().upper() for t in ticker_input.strip().split('\n') if t.strip()]
    if len(tickers) < 2:
        st.error("請輸入至少 2 檔股票")
        st.stop()
    if len(tickers) > 10:
        st.error("最多 10 檔股票")
        st.stop()

    years_history = st.slider("歷史資料年數", 3, 10, 5)
    mc_simulations = st.slider("蒙地卡羅模擬次數", 500, 5000, 1000)
    mc_years = st.slider("蒙地卡羅預測年數", 1, 5, 3)
    use_demo_data = st.checkbox("使用示範資料（無法取得即時股價時勾選）", value=False)

def generate_demo_prices(tickers: list, years: int) -> pd.DataFrame:
    """Generate realistic sample price data for demo when Yahoo API unavailable"""
    np.random.seed(42)
    n_days = int(years * 252)
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='B')
    prices_dict = {}
    base_prices = [500, 800, 120]  # 台股常見區間
    for i, t in enumerate(tickers):
        base = base_prices[i % len(base_prices)]
        daily_ret = np.random.normal(0.0005, 0.015, n_days)
        prices = base * np.exp(np.cumsum(daily_ret))
        prices_dict[t] = prices
    return pd.DataFrame(prices_dict, index=dates)

@st.cache_data(ttl=3600)
def fetch_stock_data(tickers: list, years: int) -> pd.DataFrame:
    """Fetch historical price data - fetches each ticker individually for reliability"""
    end = datetime.now()
    start = end - timedelta(days=years * 365)
    dfs = []

    for t in tickers:
        try:
            data = yf.download(t, start=start, end=end, progress=False, auto_adjust=True, threads=False, timeout=10)
            if data.empty or len(data) < 10:
                data = yf.Ticker(t).history(start=start, end=end, auto_adjust=True)
            if data.empty or len(data) < 10:
                continue
            close = data['Close'].copy() if 'Close' in data.columns else data.iloc[:, 3]
            close.name = t
            dfs.append(close)
        except Exception:
            continue

    if not dfs:
        return pd.DataFrame()

    result = pd.concat(dfs, axis=1, join='inner')
    result = result.dropna(how='all').ffill().bfill().dropna()
    return result

# Fetch data
if use_demo_data:
    with st.spinner("使用示範資料..."):
        prices = generate_demo_prices(tickers, years_history)
    st.info("目前使用示範模擬資料，非真實股價。")
else:
    with st.spinner("正在取得股價資料..."):
        prices = fetch_stock_data(tickers, years_history)

if prices.empty or len(prices) < 30:
    if not use_demo_data:
        st.warning("無法取得即時股價。改用示範資料以展示系統功能。")
        prices = generate_demo_prices(tickers, years_history)
        st.info("目前使用示範模擬資料，非真實股價。請在左側勾選「使用示範資料」可略過即時抓取。")
    else:
        st.error("示範資料產生失敗")
        st.stop()

returns = get_returns_matrix(prices)

# Optimize for 3 risk levels (needed for all tabs)
risk_levels = {'保守型': 'conservative', '平衡型': 'moderate', '積極型': 'aggressive'}
results = {}
for label, level in risk_levels.items():
    results[label] = optimize_portfolio(returns, level)

# Tab 導航
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "⓵ 最佳配置", "⓶ 未來報酬", "⓷ 五年回溯",
    "⓸ 壓力測試", "⓹ 蒙地卡羅", "⓺ 歷史股價", "⓻ 相關性矩陣"
])

with tab1:
    st.header("三種風險等級最佳配置")
    cols = st.columns(3)
    for i, (label, res) in enumerate(results.items()):
        with cols[i]:
            st.markdown(f"### {label}")
            st.metric("預期年報酬率", f"{res.expected_return*100:.2f}%")
            st.metric("年化波動率", f"{res.volatility*100:.2f}%")
            st.metric("夏普比率", f"{res.sharpe_ratio:.2f}")
            alloc_df = pd.DataFrame({'股票': res.stock_tickers, '權重': (res.weights * 100).round(1)})
            fig_pie = px.pie(alloc_df, values='權重', names='股票', title=f'{label} 配置比例')
            # st.plotly_chart(fig_pie, use_container_width=True)
            st.plotly_chart(fig_pie, width="stretch")
            st.caption("**依此配置的投資金額：**")
            for ticker, w in zip(res.stock_tickers, res.weights):
                st.caption(f"{ticker}: {principal * w:,.0f} ({w*100:.1f}%)")

# with tab2:
#     st.header("推估未來報酬率 (1-3年)")
#     for label, res in results.items():
#         proj = project_returns(returns, res.weights, 3)
#         with st.expander(f"📈 {label} 配置 - 預期報酬"):
#             c1, c2, c3 = st.columns(3)
#             c1.metric("1年後", f"{proj['1年']:.1f}%", "")
#             c2.metric("2年後", f"{proj['2年']:.1f}%", "")
#             c3.metric("3年後", f"{proj['3年']:.1f}%", "")
#             st.caption("基於歷史平均報酬的簡單複利推估，實際結果會因市場變化而不同")
with tab2:
    st.header("推估未來報酬率 (1-3年)")
    
    icons = {
        "保守型": "♟",  
        "平衡型": "♞", 
        "積極型": "♚"  
    }
    
    for label, res in results.items():
        proj = project_returns(returns, res.weights, 3)

        current_icon = next((v for k, v in icons.items() if k in label), "")

        with st.expander(f"{current_icon} {label}"):
            c1, c2, c3 = st.columns(3)
            c1.metric("1年後", f"{proj['1年']:.1f}%", "")
            c2.metric("2年後", f"{proj['2年']:.1f}%", "")
            c3.metric("3年後", f"{proj['3年']:.1f}%", "")
            st.caption("基於歷史平均報酬的簡單複利推估，實際結果會因市場變化而不同")

with tab3:
    st.header("五年回溯測試")
    st.caption("若五年前依此配置買入，現在價值多少？")
    backtest_cols = st.columns(3)
    for i, (label, res) in enumerate(results.items()):
        with backtest_cols[i]:
            bt = backtest(prices, res.weights, principal, 5)
            if 'error' in bt:
                st.warning(bt['error'])
            else:
                st.markdown(f"**{label}**")
                st.metric("現在價值", f"${bt['final_value']:,.0f}", f"{bt['cumulative_return_pct']:.1f}%")
                st.metric("總獲利", f"${bt['profit']:,.0f}", f"CAGR {bt['cagr']*100:.1f}%")
                st.caption(f"期間：{bt['start_date']} ~ {bt['end_date']}")

with tab4:
    st.header("壓力測試 - 歷史金融風暴模擬")
    st.caption("若遇到歷史上重大金融危機，組合可能虧損多少？")
    stress_results = {label: stress_test(prices, res.weights, principal) for label, res in results.items()}
    stress_data = []
    for port_label, crises in stress_results.items():
        for crisis_name, data in crises.items():
            stress_data.append({
                '配置': port_label, '危機事件': crisis_name, '期間報酬率': f"{data['return_pct']:.1f}%",
                '期末價值': data['final_value'], '虧損': data['loss']
            })
    if stress_data:
        stress_df = pd.DataFrame(stress_data)
        fig_stress = px.bar(stress_df, x='危機事件', y='虧損', color='配置', barmode='group',
            title='各危機期間預估虧損', color_discrete_map={'保守型': '#2ecc71', '平衡型': '#f39c12', '積極型': '#e74c3c'})
        fig_stress.update_layout(xaxis_tickangle=-45)
        # st.plotly_chart(fig_stress, use_container_width=True)
        st.plotly_chart(fig_stress, width="stretch")
        # st.dataframe(stress_df, use_container_width=True, hide_index=True)
        st.dataframe(stress_df, width="stretch", hide_index=True)

with tab5:
    st.header("蒙地卡羅模擬")
    st.caption(f"以 {mc_simulations} 次模擬預測 {mc_years} 年後的投資組合價值分布")
    mc_t1, mc_t2, mc_t3 = st.tabs(["保守型", "平衡型", "積極型"])
    for tab, (label, res) in zip([mc_t1, mc_t2, mc_t3], results.items()):
        with tab:
            paths, stats = monte_carlo_simulation(returns, res.weights, principal, mc_years, mc_simulations)
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown("**模擬統計結果**")
                st.metric("平均價值", f"${stats['mean']:,.0f}")
                st.metric("中位數", f"${stats['median']:,.0f}")
                st.metric("5% 分位（較差）", f"${stats['percentile_5']:,.0f}")
                st.metric("95% 分位（較佳）", f"${stats['percentile_95']:,.0f}")
            with col2:
                fig_mc = make_subplots(rows=2, cols=1, subplot_titles=('模擬路徑 (抽樣100條)', '期末價值分布'), vertical_spacing=0.22)
                n_show = min(100, mc_simulations)
                for i in range(0, n_show, max(1, n_show // 20)):
                    fig_mc.add_trace(go.Scatter(y=paths[i], mode='lines', line=dict(width=1, color='rgba(100,150,255,0.3)')), row=1, col=1)
                fig_mc.add_histogram(x=paths[:, -1], nbinsx=50, marker_color='#636efa', row=2, col=1)
                fig_mc.update_layout(height=650, showlegend=False, margin=dict(t=50, b=50))#新增, margin=dict(t=50, b=50).原height=500
                fig_mc.update_xaxes(title_text="交易日", row=1, col=1)
                fig_mc.update_xaxes(title_text="期末價值", row=2, col=1)
                # st.plotly_chart(fig_mc, use_container_width=True)
                st.plotly_chart(fig_mc, width="stretch")
                

with tab6:
    st.header("歷史股價走勢")
    norm_prices = (prices / prices.iloc[0] * 100)
    fig_price = px.line(norm_prices, title='標準化股價走勢 (基期=100)')
    fig_price.update_layout(yaxis_title="指數", xaxis_title="日期", legend_title="股票")
    # st.plotly_chart(fig_price, use_container_width=True)
    st.plotly_chart(fig_price, width="stretch")

with tab7:
    st.header("股票相關性矩陣")
    corr = returns.corr()
    fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale='RdBu_r', aspect='auto')
    fig_corr.update_layout(title='報酬率相關係數')
    # st.plotly_chart(fig_corr, use_container_width=True)
    st.plotly_chart(fig_corr, width="stretch")

st.success("以上結果僅供參考，投資有風險，請審慎評估。")
