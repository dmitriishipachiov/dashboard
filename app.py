import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from moexalgo import Ticker
from datetime import datetime, timedelta

# ==========================================
# 1. ГЕНЕРАЦИЯ ДАННЫХ (Имитация MOEX)
# ==========================================
def get_moex_data(ticker: str, days: int=365):
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days)
    t = Ticker(ticker)

    df = t.candles(
        start=start_date,
        end=end_date,
        period='1D'
    )

    df['begin'] = pd.to_datetime(df['begin'])
    df.set_index('begin', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']]
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

    return df

tickers = ['SBER', 'GAZP', 'LKOH']

mdd_comparison = {} # Здесь будем хранить {Тикер: Просадка}

for t in tickers:
    try:
        temp_df = get_moex_data(t)
        # Считаем доходность и просадку для каждого
        temp_returns = temp_df['Close'].pct_change().fillna(0)
        temp_equity = (1 + temp_returns).cumprod()
        temp_mdd = (temp_equity / temp_equity.cummax() - 1).min()
        mdd_comparison[t] = temp_mdd
    except Exception as e:
        mdd_comparison[t] = -1.0 # Если ошибка, ставим плохую просадку

# Находим тикер с самой маленькой просадкой (максимальное значение, так как они отрицательные)
best_asset = max(mdd_comparison, key=mdd_comparison.get)
best_mdd_value = mdd_comparison[best_asset] * 100

# ==========================================
# 2. ИНТЕРФЕЙС DASHBOARD (Streamlit)
# ==========================================

# Настройка страницы
st.set_page_config(page_title="QuantAnalytica MOEX", layout="wide")

# Боковая панель (Sidebar)
st.sidebar.title("⚙️ Настройки анализа")
selected_ticker = st.sidebar.selectbox("Выберите акцию:", tickers)
window_short = st.sidebar.slider("Окно скользящей средней (SMA):", 3, 50, 20)
window_long = st.sidebar.slider("Окно скользящей средней (LSMA):", 50, 100, 200)
show_drawdown = st.sidebar.checkbox("Показать график просадки", value=True)

# Заголовок главного экрана
st.title(f"📊 Аналитический отчет по активу: {selected_ticker}")
st.markdown(f"Этот дашборд генерирует автоматический анализ на основе математических моделей.")

# Загрузка данных
df = get_moex_data(selected_ticker)

# Расчеты (то, что ты уже умеешь!)
df['SMA'] = df['Close'].rolling(window=window_short).mean()
df['LSMA'] = df['Close'].rolling(window=window_long).mean()
df['returns'] = df['Close'].pct_change()
df['equity'] = (1 + df['returns'].fillna(0)).cumprod()

# Расчет просадки
rolling_max = df['equity'].cummax()
df['drawdown'] = (df['equity'] / rolling_max) - 1

# ==========================================
# 3. ВИЗУАЛИЗАЦИЯ МЕТРИК (Карточки сверху)
# ==========================================
col1, col2, col3 = st.columns(3)

with col1:
    last_price = df['Close'].iloc[-1]
    st.metric("Текущая цена", f"{last_price:.2f} ₽", f"{df['returns'].iloc[-1]*100:.2f}%")

with col2:
    total_return = (df['equity'].iloc[-1] - 1) * 100
    st.metric("Общая доходность", f"{total_return:.2f}%")

with col3:
    max_dd = df['drawdown'].min() * 100
    st.metric("Макс. просадка (MDD)", f"{max_dd:.2f}%", delta_color="inverse")

# ==========================================
# 4. ГРАФИКИ
# ==========================================
fig, ax = plt.subplots(2 if show_drawdown else 1, 1, figsize=(10, 8), sharex=True)

if show_drawdown:
    # Верхний график: Цена и SMA
    ax[0].plot(df.index, df['Close'], label='Цена', color='blue')
    ax[0].plot(df.index, df['SMA'], label=f'SMA {window_short}', color='orange', linewidth=2)
    ax[0].plot(df.index, df['LSMA'], label=f'LSMA {window_long}', color='red', linewidth=2)
    ax[0].set_title(f"Динамика цены {selected_ticker}")
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    # Нижний график: Просадка
    ax[1].fill_between(df.index, df['drawdown'], 0, color='red', alpha=0.3)
    ax[1].set_title("График просадки (Equity Drawdown)")
    ax[1].set_ylabel("Падение от пика")
    ax[1].grid(True, alpha=0.3)
else:
    ax.plot(df.index, df['Close'], label='Цена', color='blue')
    ax.plot(df.index, df['SMA'], label=f'SMA {window_short}', color='orange', linewidth=2)
    ax.plot(df.index, df['LSMA'], label=f'LSMA {window_long}', color='red', linewidth=2)
    ax.set_title(f"Динамика цены {selected_ticker}")
    ax.legend()
    ax.grid(True, alpha=0.3)

st.pyplot(fig)

# ==========================================
# 5. ТАБЛИЦА АНАЛИТИКИ (Для продажи)
# ==========================================
st.subheader("📋 Сравнительная таблица активов")
comparison_list = []
for t in tickers:
    comparison_list.append({
        'Тикер': t, 
        'Макс. Просадка (%)': f"{mdd_comparison[t]*100:.2f}%", 
        'Статус': 'Устойчив' if mdd_comparison[t] > -0.2 else 'Волатилен'
    })

st.table(pd.DataFrame(comparison_list))

# АКТУАЛЬНЫЙ СОВЕТ
st.info(f"💡 Совет: Акции **{best_asset}** сейчас показывают лучшую устойчивость к просадкам " 
        f"среди выбранных активов (Max DD: {best_mdd_value:.2f}%). Это делает их более привлекательными "
        f"для консервативного инвестора в текущем цикле.")