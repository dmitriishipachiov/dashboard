import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from moexalgo import Ticker
from datetime import datetime, timedelta

# ==========================================
# 1. ПОЛУЧЕНИЕ ДАННЫХ (MOEX)
# ==========================================
def get_moex_data(ticker: str, days: int=365):
    try:
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days)
        t = Ticker(ticker)
        df = t.candles(start=start_date, end=end_date, period='1D')
        
        if df is None or df.empty:
            return None

        df['begin'] = pd.to_datetime(df['begin'])
        df.set_index('begin', inplace=True)
        df = df[['open', 'high', 'low', 'close', 'volume']]
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        return df
    except Exception as e:
        return None

def load_tickers(file_path):
    try:
        with open(file_path, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        return ['SBER', 'GAZP', 'LKOH']

# ==========================================
# 2. ЯДРО АНАЛИТИКИ (Математика)
# ==========================================
def calculate_strategy(df, short_w, long_w, use_rsi=False):
    tdf = df.copy()
    
    # 1. Индикаторы
    tdf['SMA_s'] = tdf['Close'].rolling(window=short_w).mean()
    tdf['SMA_l'] = tdf['Close'].rolling(window=long_w).mean()
    tdf['returns'] = tdf['Close'].pct_change().fillna(0)
    
    # 2. RSI
    delta = tdf['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    tdf['RSI'] = 100 - (100 / (1 + rs))

    # 3. Сигналы
    if use_rsi:
        # Условие: Пересечение ВВЕРХ + RSI ниже 70 (не перекуплено)
        condition = (tdf['SMA_s'] > tdf['SMA_l']) & (tdf['RSI'] < 70)
    else:
        condition = (tdf['SMA_s'] > tdf['SMA_l'])

    tdf['signal'] = condition.astype(int).shift(1).fillna(0)
    
    # 4. Расчеты доходности и просадки
    tdf['strat_ret'] = tdf['returns'] * tdf['signal']
    tdf['equity_strat'] = (1 + tdf['strat_ret']).cumprod()
    tdf['equity_mkt'] = (1 + tdf['returns']).cumprod()
    
    tdf['peak'] = tdf['equity_strat'].cummax()
    tdf['drawdown'] = (tdf['equity_strat'] / tdf['peak']) - 1
    
    total_return_val = float(tdf['equity_strat'].iloc[-1] - 1)
    max_dd_val = float(tdf['drawdown'].min())
    
    return total_return_val, max_dd_val, tdf

# ==========================================
# 3. ИНТЕРФЕЙС DASHBOARD
# ==========================================
st.set_page_config(page_title="QuantAnalytica Pro", layout="wide")

tickers_list = load_tickers('tickers.txt')

st.sidebar.title("⚙️ Настройки")
selected_tickers = st.sidebar.multiselect("Выберите акции:", tickers_list, default=tickers_list[:1])
use_rsi_filter = st.sidebar.checkbox("Использовать RSI фильтр", value=True)

w_short = st.sidebar.slider("SMA Short", 5, 50, 20)
w_long = st.sidebar.slider("SMA Long", 50, 200, 100)

run_opt = st.sidebar.button("🚀 Оптимизировать актив")

if not selected_tickers:
    st.warning("Выберите акцию.")
    st.stop()

all_data = {}
for t in selected_tickers:
    data = get_moex_data(t)
    if data is not None:
        all_data[t] = data

if not all_data:
    st.error("Ошибка загрузки данных.")
    st.stop()

main_ticker = selected_tickers[0]
res_ret, res_mdd, df_plot = calculate_strategy(all_data[main_ticker], w_short, w_long, use_rsi_filter)

st.title(f"📊 Аналитика: {main_ticker}")

# --- МЕТРИКИ ---
col1, col2, col3 = st.columns(3)
with col1:
    st_ret_pct = res_ret * 100
    st.metric("Доходность стратегии", f"{st_ret_pct:.2f}%")
with col2:
    st_mdd_pct = res_mdd * 100
    st.metric("Max Drawdown", f"{st_mdd_pct:.2f}%", delta_color="inverse")
with col3:
    vol = df_plot['returns'].std() * np.sqrt(252) * 100
    st.metric("Волатильность (год)", f"{vol:.2f}%")

# --- ГРАФИКИ (3 ПОДГРАФИКА) ---
st.subheader(f"🔍 Глубокий анализ: {main_ticker}")
fig, ax = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

# 1. Цена и SMA
ax[0].plot(df_plot.index, df_plot['Close'], label="Цена Close", color='blue', alpha=0.6)
ax[0].plot(df_plot.index, df_plot['SMA_s'], label=f'SMA {w_short}', color='orange', linewidth=2)
ax[0].plot(df_plot.index, df_plot['SMA_l'], label=f'SMA {w_long}', color='red', linewidth=2)
ax[0].set_title("1. Движение цены и пересечение средних")
ax[0].legend(loc='upper left')
ax[0].grid(True, alpha=0.3)

# 2. RSI
ax[1].plot(df_plot.index, df_plot['RSI'], label='RSI', color='purple')
ax[1].axhline(70, color='red', linestyle='--', alpha=0.5) # Зона перекупленности
ax[1].axhline(30, color='green', linestyle='--', alpha=0.5) # Зона перепроданности
ax[1].fill_between(df_plot.index, 70, 100, color='red', alpha=0.05) # Заливка сверху
ax[1].fill_between(df_plot.index, 0, 30, color='green', alpha=0.05) # Заливка снизу
ax[1].set_title("2. Индекс RSI (Фильтр силы тренда)")
ax[1].set_ylim(0, 100)
ax[1].legend(loc='upper left')
ax[1].grid(True, alpha=0.3)

# 3. Equity Curve
ax[2].plot(df_plot.index, df_plot['equity_strat'], label="Стратегия (Net)", color='green', linewidth=2)
ax[2].plot(df_plot.index, df_plot['equity_mkt'], label="Рынок (B&H)", color='gray', linestyle='--', alpha=0.6)
ax[2].set_title("3. Накопленная доходность (Equity Curve)")
ax[2].legend(loc='upper left')
ax[2].grid(True, alpha=0.3)

st.pyplot(fig)

# --- ОПТИМИЗАЦИЯ И СРАВНЕНИЕ (как было) ---
tab1, tab2, tab3 = st.tabs(["🔍 Оптимизация", "🌍 Сравнение", "📋 Данные"])

with tab1:
    if run_opt:
        st.subheader(f"Оптимизация для {main_ticker}")
        short_range = range(5, 41, 5)
        long_range = range(50, 151, 20)
        heatmap_data = np.zeros((len(short_range), len(long_range)))

        with st.spinner('Поиск...'):
            for i, s in enumerate(short_range):
                for j, l in enumerate(long_range):
                    if s < l:
                        r, _, _ = calculate_strategy(all_data[main_ticker], s, l, use_rsi_filter)
                        heatmap_data[i, j] = r
                    else:
                        heatmap_data[i, j] = -999

        fig_hm, ax_hm = plt.subplots(figsize=(10, 6))
        sns.heatmap(heatmap_data, annot=True, fmt=".1%", xticklabels=long_range, yticklabels=short_range, cmap="RdYlGn", ax=ax_hm)
        st.pyplot(fig_hm)
        
        idx_max = np.unravel_index(np.argmax(heatmap_data), heatmap_data.shape)
        st.success(f"✅ Лучшая пара: SMA {short_range[idx_max[0]]} / SMA {long_range[idx_max[1]]}")
    else:
        st.info("Нажмите 'Оптимизировать' для запуска Grid Search.")

with tab2:
    st.subheader("Сравнение всех выбранных активов")
    fig_comp, ax_comp = plt.subplots(figsize=(12, 5))
    for t in all_data.keys():
        temp_df = all_data[t].copy()
        temp_df['equity'] = (1 + temp_df['Close'].pct_change().fillna(0)).cumprod()
        ax_comp.plot(temp_df.index, temp_df['equity'], label=t)
    ax_comp.legend()
    ax_comp.grid(alpha=0.3)
    st.pyplot(fig_comp)

with tab3:
    st.subheader("Таблица последних цен")
    st.dataframe(all_data[main_ticker].tail(10))
