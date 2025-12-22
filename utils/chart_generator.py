import streamlit as st
import plotly.express as px
import pandas as pd


class ChartGenerator:
    def correlation_heatmap(self, df: pd.DataFrame):
        corr = df.corr(numeric_only=True)
        fig = px.imshow(corr, aspect="auto", color_continuous_scale="RdBu_r")
        st.plotly_chart(fig, use_container_width=True)

    def histogram(self, df: pd.DataFrame, col: str, color_col=None, marginal=None):
        try:
            fig = px.histogram(df, x=col, color=color_col, marginal=marginal)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error generating histogram: {e}")

    def box_plot(self, df: pd.DataFrame, x_col=None, y_col=None):
        try:
            if y_col is None:
                st.warning("Please select a Y-axis column.")
                return
            if x_col:
                fig = px.box(df, x=x_col, y=y_col, color=x_col)
            else:
                fig = px.box(df, y=y_col)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error generating box plot: {e}")
