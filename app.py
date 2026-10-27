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
        
        df['begin'] = pd.to_datetime(df['begin'])
        df.set_index('begin', inplace=True)
        df = df[['open', 'high', 'low', 'close', 'volume']]
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        return df
    except Exception as e:
        st.error(f"Ошибка загрузки данных: {e}")
        return None

# ==========================================
# 2. ЯДРО АНАЛИТИКИ (Математика)
# ==========================================
def calculate_strategy(df, short_w, long_w, use_rsi=False):
    """Вычисляет доходность стратегии для заданных параметров"""
    temp_df = df.copy()
    temp_df['SMA_s'] = temp_df['Close'].rolling(window=short_w).mean()
    temp_df['SMA_l'] = temp_df['Close'].rolling(window=long_w).mean()
    temp_df['returns'] = temp_df['Close'].pct_change().fillna(0)
    
    # RSI
    delta = temp_df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    temp_df['RSI'] = 100 - (100 / (1 + rs))

    if use_rsi:
        # Условие: Пересечение + RSI < 70
        signal = (temp_df['SMA_s'] > temp_df['SMA_l']) & (temp_df['RSI'] < 70)
    else:
        # Простое пересечение
        signal = (temp_df['SMA_s'] > temp_df['LSMA']) if 'LSMA' in temp_df else (temp_df['SMA_s'] > temp_df['SMA_l'])
        # Исправлено для логики:
        signal = (temp_df['SMA_s'] > temp_df['SMA_l'])

    temp_df['signal'] = signal.astype(int).shift(1).fillna(0)
    temp_df['strat_ret'] = temp_df['returns'] * temp_df['signal']
    total_return = (1 + temp_df['strat_ret']).cumprod().iloc[-1] - 1
    return total_return, temp_df

# ==========================================
# 3. ИНТЕРФЕЙС DASHBOARD (Streamlit)
# ==========================================
st.set_page_config(page_title="QuantAnalytica Pro", layout="wide")

st.sidebar.title("⚙️ Настройки")
selected_ticker = st.sidebar.selectbox("Выберите акцию:", ['SBER', 'GAZP', 'LKOH'])
use_rsi_filter = st.sidebar.checkbox("Использовать RSI фильтр", value=True)

st.title(f"🚀 Анализ стратегии: {selected_ticker}")

# Загрузка данных
df = get_moex_data(selected_ticker)

if df is not None:
    # --- Боковая панель управления ---
    st.sidebar.subheader("Параметры стратегии")
    w_short = st.sidebar.slider("SMA Short", 5, 50, 20)
    w_long = st.sidebar.slider("SMA Long", 50, 200, 100)
    
    st.sidebar.divider()
    st.sidebar.subheader("Оптимизация (Grid Search)")
    run_opt = st.sidebar.button("🚀 Запустить поиск лучших параметров")

    # --- Основные расчеты (текущие параметры) ---
    res, df_plot = calculate_strategy(df, w_short, w_long, use_rsi_filter)
    
    # Вывод метрик
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Текущая доходность", f"{res*100:.2f}%")
    with col2:
        mdd = (df_plot['equity'].cummax() / df_plot['equity'] - 1).max() if 'equity' in locals() else 0
        # Пересчитаем правильно для текущего df
        equity = (1 + df_plot['strat_ret']).cumprod()
        mdd = (equity.cummax() / equity - 1).max() * 100
        st.metric("Max Drawdown", f"-{mdd:.2f}%", delta_color="inverse")
    with col3:
        st.metric("Волатильность", f"{df['Close'].pct_change().std()*100:.2f}%")

    # --- ГРАФИК ---
    st.subheader("📈 График доходности")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df_plot.index, (1 + df_plot['strat_ret']).cumprod(), label="Стратегия", color='green')
    ax.plot(df_plot.index, (1 + df_plot['returns']).cumprod(), label="Рынок (Buy & Hold)", color='gray', linestyle='--')
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)

    # --- БЛОК ОПТИМИЗАЦИИ (GRID SEARCH) ---
    if run_opt:
        st.subheader("🔍 Поиск оптимальных параметров (Heatmap)")
        
        short_range = range(5, 41, 5)
        long_range = range(50, 151, 20)
        
        # Матрица результатов
        heatmap_data = np.zeros((len(short_range), len(long_range)))

        with st.spinner('Идет перебор параметров... Пожалуйста, подождите.'):
            for i, s in enumerate(short_range):
                for j, l in enumerate(long_range):
                    if s < l:
                        ret, _ = calculate_strategy(df, s, l, use_rsi_filter)
                        heatmap_data[i, j] = ret
                    else:
                        heatmap_data[i, j] = -999 # Некорректные параметры

        # Визуализация
        fig_hm, ax_hm = plt.subplots(figsize=(10, 7))
        sns.heatmap(heatmap_data, annot=True, fmt=".2%", 
                    xticklabels=long_range, yticklabels=short_range,
                    cmap="RdYlGn", ax=ax_hm)
        ax_hm.set_xlabel("SMA Long (Длинная)")
        ax_hm.set_ylabel("SMA Short (Короткая)")
        st.pyplot(fig_hm)
        
        # Поиск лучшего
        idx_max = np.unravel_index(np.argmax(heatmap_data), heatmap_data.shape)
        best_s = short_range[idx_max[0]]
        best_l = long_range[idx_max[1]]
        
        st.success(f"✅ Лучшая комбинация: **SMA Short: {best_s}** и **SMA Long: {best_l}**")
        st.warning("⚠️ ВНИМАНИЕ: Если на тепловой карте видна одна яркая точка среди красных — это OVERFITTING (подгонка). Ищите 'плато' (область стабильно зеленого цвета).")
