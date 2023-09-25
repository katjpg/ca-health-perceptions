from dash import Dash, dcc, html, Input, Output, callback_context
import plotly.express as px
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
import json
import statsmodels.api as sm


# --------------- LOAD DATA ---------------
# load the county health data and filter for the year 2022
df = pd.read_csv('data/processed/county-health-final.csv')
df = df[df['year'] == 2022]

# load the geojson data for the county boundaries
with open('data/processed/ca-counties.geojson') as geo:
    counties = json.load(geo)
    
# --------------- GLOBAL VARIABLES ---------------
# define the indicator variables for different categories
SOCIO_VAR = ['pct_uninsured', 'pct_children_poverty', 'median_hh_income', 'pct_hs_complete', 'pct_some_college', 
             'pct_unemployed', 'pct_single_parent_hous']

ENV_VAR = ['food_env_index', 'avg_daily_pm25', 'pct_severe_hous_prob', 'severe_hous_cost_burden', 'pct_overcrowding', 
           'pct_limit_access_health_food']

HEALTH_VAR = ['pct_fair_poor_health', 'pct_smokers', 'pct_obesity', 'pct_excess_drinking', 'prim_care_phys_rate', 
              'mental_health_prov_rate', 'preventable_rate', 'pct_freq_ment_distress', 'pct_freq_phys_distress', 
              'pct_diabetes', 'pct_food_insecure', 'pct_drug_overdose_mortality']

RANK_VAR = ['health_outcome_rank', 'health_fac_rank', 'length_life_rank', 'quality_life_rank', 'health_behaviors_rank', 
            'clinical_care_rank']

# combine all variables
ALL_VAR = SOCIO_VAR + ENV_VAR + HEALTH_VAR

# define the features for cluster analysis
FEATURES = ['pct_freq_phys_distress', 'pct_some_college', 'pct_freq_ment_distress', 'pct_obesity', 'pct_smokers', 
            'pct_children_poverty', 'pct_unemployed', 'pct_uninsured', 'prim_care_phys_rate', 'median_hh_income', 
            'preventable_rate', 'pct_overcrowding', 'pct_diabetes', 'pct_excess_drinking', 'pct_food_insecure']

# define the colors for the clusters
COLORS = ["#003459", "#028090", "#02c39a"] 

# --------------- APP LAYOUT ---------------
# initialize the Dash app with the Flatly Bootstrap theme
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])

# define the header of the app
header = html.H3(
    "Perceptions of Health Across California Counties (2022)",
    className="bg-primary text-white p-2 mb-2 text-center"
)

# define the header for the data exploration section
header_eda = html.H3(
    "Exploring the Data",
    className="text-center"
)

# define the dropdown for selecting the category of indicators
dropdown_category = dbc.Row([
    dbc.Col([
        dbc.Label("Select a category:"),
        dcc.RadioItems(
            id='indicator-category',
            options=[
                {'label': 'Socioeconomic', 'value': 'soc_eco'},
                {'label': 'Environmental', 'value': 'env'},
                {'label': 'Health', 'value': 'health'}
            ],
            value='soc_eco'
        )
    ])
], className="mb-4")

# define the dropdown for selecting the indicator variable
dropdown_indicator = dbc.Row([
    dbc.Col([
        dbc.Label("Select an indicator:"),
        dcc.Dropdown(
            id='indicator-dpdn',
            options=[],
            multi=False,
            placeholder='Select an Indicator'
        )
    ])
], className="mb-4")

# define the dropdown for selecting the variables for the heatmap
heatmap_dropdown = dbc.Row([
    dbc.Col([
        dbc.Label("Select up to 8 variables for correlation heatmap:"),
        dcc.Dropdown(
            id='heatmap-variable-dpdn',
            options=[{'label': var, 'value': var} for var in ALL_VAR],
            multi=True,
            placeholder='Select Variables'
        ),
        html.Div(id='limit-warning', style={'color': 'red'})
    ])
], className="mb-4")

# define the control panel for the app
controls = dbc.Card(
    [dropdown_category, dropdown_indicator, heatmap_dropdown],
    body=True,
)

# define the tabs for the indicator map and correlation heatmap
tab1 = dbc.Tab([
    dbc.Spinner([
        dcc.Graph(id='indicator-map')
    ])
], label="Indicator Map")

tab2 = dbc.Tab([dcc.Graph(id='corr-map')], label="Correlation Heatmap")

# define the dropdown for selecting the health ranking variable for the scatterplot
ranking_dropdown = html.Div(
    [
        dbc.Label("Select Ranking:"),
        dcc.Dropdown(
            id='ranking',
            options=[{'label': i, 'value': i} for i in RANK_VAR],
            value=RANK_VAR[0],  # default ranking
            clearable=False,
        )
    ],
    className='mb-4',
)

# define the scatterplot for the app
scatterplot = dcc.Graph(id='scatterplot')

# define the card for the scatterplot and its control
scatterplot_card = dbc.Card(
    [
        dbc.CardBody(
            [
                ranking_dropdown,
                scatterplot,
            ],
        ),
    ],
)

# define the card for the description of the scatterplot
description_card = dbc.Card(
    [
        dbc.CardHeader("What's this?"),
        dbc.CardBody(
            "The scatterplot contrasts perceived health status "
            "(pct_poor_fair_health) with selected county health rankings. "
            "It reveals how subjective health perceptions align with objective "
            "health indicators. Users can select different health rankings for personalized data exploration."
        ),
    ],
)

# define the card for the limitations of the scatterplot
limitations_card = dbc.Card(
    [
        dbc.CardHeader("What are the limitations?"),
        dbc.CardBody(
            [
                html.P("While the County Health Rankings (CHR) offer pivotal insights into health disparities at the county level, "
                       "their scope may not fully capture the multifaceted factors influencing self-perceived health status."),
                
                html.P("Individual health perceptions are shaped by a complex interplay of factors, many of which stretch "
                       "beyond the conventional socioeconomic and demographic parameters utilized by CHR rankings.")
            ]
        ),
    ],
    className="mt-4",
)

# define the header for the correlation analysis section
header_correlation = html.H3(
    "Analyzing Correlations Between Perceived Health Status and Indicators", 
    className="mt-4 text-center"
)

# define the information text for the correlation analysis section
correlation_info_text = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.P(
                            "To gain a more holistic understanding of potential predictors "
                            "influencing the target variable, I incorporated both ranked measure "
                            "data and additional measure data in my analysis."
                            " This comprehensive approach strives to encapsulate both the "
                            "subjective nuances of personal health perceptions and the objective "
                            "health determinants emphasized in CHR, offering a more robust and "
                            "nuanced perspective on health disparities."
                        ),
                    ]
                ),
            ]
        ),
    ],
    className="text-center mt-4",
)

# define the dropdown for selecting the predictor variable for the scatterplot with marginals
dropdown_predictor = dbc.Row(
    [
        dbc.Col(
            [
                dbc.Label("Select a Predictor Variable:"),
                dcc.Dropdown(
                    id='predictor-dpdn',
                    options=[{'label': i, 'value': i} for i in ALL_VAR],
                    value=ALL_VAR[0],  # Default value
                    multi=False,
                    placeholder='Select a Predictor Variable'
                )
            ]
        )
    ],
    className="mb-4",
)

# define the scatterplot with marginals for the app
scatter_with_marginals = dcc.Graph(id='scatter_marginal')

# define the card for the scatterplot with marginals and its control
correlation_card = dbc.Card(
    [
        dbc.CardBody(
            [
                dropdown_predictor,
                scatter_with_marginals
            ]
        ),
    ],
    className="mt-4"
)

# define the section for correlation analysis
correlation_section = html.Div(
    [
        header_correlation,
        correlation_card
    ]
)

# define the header for the cluster analysis section
header_cluster = html.H3(
    "Cluster Analysis of Counties", 
    className="mt-4 text-center"
)

# define the information text for the cluster analysis section
cluster_info_text = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.P(
                            "After conducting feature selection techniques, 15 features were "
                            "identified that provide a comprehensive representation of demographic, "
                            "health, and socioeconomic factors. This optimized dataset was used to "
                            "predict the target variable. "
                            "I then applied the K-means clustering, an unsupervised learning "
                            "algorithm, to categorize California counties based on similarities in "
                            "health perceptions, socioeconomic characteristics, health indicators, "
                            "and environmental conditions. The aim was to discern distinct features "
                            "or factors that characterize and differentiate these county groupings, "
                            "with a focus on understanding the role of socioeconomic, demographic, "
                            "health, and environmental predictors in shaping these differences."
                        ),
                    ]
                ),
            ]
        ),
    ],
    className="text-center  mt-4"
)

# define the checklist for selecting the clusters for the cluster map
cluster_checkboxes = dcc.Checklist(
    id='cluster-checklist',
    options=[
        {'label': '1', 'value': '1'},
        {'label': '2', 'value': '2'},
        {'label': '3', 'value': '3'},
    ],
    value=['1'],  # Default value
    inline=True,
)

# define the card for the cluster map and its control
cluster_card = dbc.Card(
    [
        dbc.CardBody(
            [
                cluster_checkboxes,
                dbc.Spinner(dcc.Graph(id='cluster-map')), 
            ]
        ),
    ],
    className="mt-4"
)

# define the dropdown for selecting the feature for the cluster comparison plot
features_dropdown = dbc.Row(
    [
        dbc.Col(
            [
                dbc.Label("Select a feature for cluster comparison:"),
                dcc.Dropdown(
                    id='features-dpdn',
                    options=[{'label': i, 'value': i} for i in FEATURES],
                    value=FEATURES[0],  # Default value
                    multi=False,
                    placeholder='Select a Feature:'
                )
            ]
        )
    ],
    className="mb-4"
)

# define the cluster comparison plot for the app
cluster_comparison_plot = dcc.Graph(id='cluster-comparison-plot')

# define the card for the cluster comparison plot and its control
cluster_comparison_card = dbc.Card(
    [
        dbc.CardBody(
            [
                features_dropdown,
                cluster_comparison_plot
            ]
        ),
    ],
    className="mt-4"
)

# define the layout of the app
app.layout = dbc.Container(
    [
        header,
        header_eda,
        dbc.Row(
            [
                dbc.Col(
                    [
                        controls,
                    ],
                    className="mt-4",
                    md=5,
                ),
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.Tabs(
                                    [
                                        dbc.Tab(tab1, label="Indicator Map"),
                                        dbc.Tab(tab2, label="Correlation Heatmap"),
                                    ],
                                )
                            ],
                            className="mt-4",
                        )
                    ],
                    md=7,
                ),
            ]
        ),
        dbc.Row( 
            [
                dbc.Col(
                    [
                        scatterplot_card,
                    ],
                    className="mt-4",
                    md=8,  
                    style={"margin": "15px 0"},
                ),
                dbc.Col(
                    [
                        description_card,
                        limitations_card,
                    ],
                    className="mt-4",
                    md=4,
                ),
            ]
        ),
        header_correlation,
        correlation_info_text,
        dbc.Row(
            [
                dbc.Col(
                    [
                        correlation_card,
                    ],
                ),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        header_cluster,
                        cluster_info_text,
                    ],
                ),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        cluster_card,
                    ],
                    md=6,
                ),
                dbc.Col(
                    [
                        cluster_comparison_card,
                    ],
                    md=6,
                ),
            ]
        ),        
    ],
    style={'max-width': '80%', 'margin-left': 'auto', 'margin-right': 'auto'},
    fluid=True,
    className="dbc",
)

# --------------- APP INTERACTIVITY ---------------
# updates the dropdown options based on the selected category from the radio button
@app.callback(
    Output('indicator-dpdn', 'options'),
    Input('indicator-category', 'value')
)
def update_indicator_dpdn(selected_category):
    # return the appropriate list of indicators as options for the dropdown
    if selected_category == 'soc_eco':
        return [{'label': i, 'value': i} for i in SOCIO_VAR]
    elif selected_category == 'env':
        return [{'label': i, 'value': i} for i in ENV_VAR]
    elif selected_category == 'health':
        return [{'label': i, 'value': i} for i in HEALTH_VAR]

# updates the choropleth map figure based on the selected indicator from the dropdown
@app.callback(
    Output('indicator-map', 'figure'),
    Input('indicator-dpdn', 'value')
)
def update_map(indicator):
    # create a choropleth map using the selected indicator as the color dimension
    fig = px.choropleth_mapbox(df,
                               geojson=counties,
                               color=indicator,
                               locations='county',
                               featureidkey="properties.COUNTY_NAME",
                               color_continuous_scale="emrld",
                               labels={indicator: 'Indicator'},
                               title='County Indicators',
                               mapbox_style="carto-positron",
                               zoom=5.5,
                               center={"lat": 36.7783, "lon": -119.4179},  # center on California
                               opacity=0.5,
                               hover_name='county',
                               hover_data={indicator: True}
                               )
    # remove margins around the plot
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    return fig

# updates the scatterplot figure based on the selected ranking from the dropdown
@app.callback(
    Output('scatterplot', 'figure'),
    Input('ranking', 'value')
)
def update_scatterplot(selected_ranking):
    """Update the scatterplot with selected ranking."""
    # create a scatter plot with percentage of fair/poor health on the y-axis and the selected ranking on the x-axis
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        y=df["pct_fair_poor_health"], 
        x=df[selected_ranking], 
        mode='markers', 
        marker=dict(
            size=6,
            color="lightseagreen",
        ),
        text=df["county"],
        hovertemplate = '<i>'+selected_ranking+'</i>: %{x}<br><i>Percentage of Fair/Poor Health</i>: %{y}<br><b>%{text}</b>',
        showlegend=False
    ))

    fig.update_layout(
        title="Perceived Health Status vs. County Health Rankings",
        yaxis_title="Percentage of Fair/Poor Health",
        xaxis_title=selected_ranking,
        hovermode='closest'
    )

    return fig

# limit the number of variables that can be selected in the heatmap dropdown to 8
@app.callback(
    Output('limit-warning', 'children'),
    Output('heatmap-variable-dpdn', 'value'),
    Input('heatmap-variable-dpdn', 'value')
)
def limit_heatmap_dropdown_selection(selected_vars):
    if selected_vars is None:
        return '', []
    
    # if more than 8 variables are selected, return a warning message and remove the last selected variable from the list
    if len(selected_vars) > 8:
        return 'Limit exceeded. Please select up to 8 variables.', selected_vars[:-1]
    
    return '', selected_vars

# updates the heatmap figure based on the selected variables from the dropdown
@app.callback(
    Output('corr-map', 'figure'),
    Input('heatmap-variable-dpdn', 'value')
)
def update_corr_map(selected_variables):
    # calculate the correlation matrix for the selected variables
    correlation = df[selected_variables].corr()
    # create a heatmap using the correlation matrix
    fig = go.Figure(data=go.Heatmap(
                   z=correlation,
                   x=selected_variables,
                   y=selected_variables,
                   colorscale='Sunsetdark'))
    fig.update_layout(title='Correlation Heatmap')
    return fig

# updates the scatterplot with marginal histograms figure based on the selected predictor from the dropdown
@app.callback(
    Output('scatter_marginal', 'figure'),
    Input('predictor-dpdn', 'value')
)
def update_scatter_hist(selected_indicator):
    # fit an OLS model using the selected predictor as the independent variable and percentage of fair/poor health as the dependent variable
    X = df[selected_indicator]
    y = df["pct_fair_poor_health"]

    X = sm.add_constant(X)
    model = sm.OLS(y, X)
    results = model.fit()
    
    y_pred = results.predict(X)
    
    # divide the data into two groups based on whether the observed y value is above or below the predicted y value
    above = df[y > y_pred]
    below = df[y <= y_pred]

    fig = go.Figure()
    
    # add a scatter plot trace for each group to the figure
    fig.add_trace(go.Scatter(
        text=above["county"],
        x=above[selected_indicator], 
        y=above["pct_fair_poor_health"],
        mode='markers', 
        name='Above Average',
        marker=dict(color='lightseagreen'),
        hovertemplate = '<i>'+selected_indicator+'</i>: %{x}<br><i>Percentage of Fair/Poor Health</i>: %{y}<br><b>%{text}</b>'
    ))
    
    fig.add_trace(go.Scatter(
        x=below[selected_indicator], 
        y=below["pct_fair_poor_health"],
        text=below["county"],
        mode='markers', 
        name='Below Average',
        marker=dict(color='mediumaquamarine'),
        hovertemplate = '<i>'+selected_indicator+'</i>: %{x}<br><i>Percentage of Fair/Poor Health</i>: %{y}<br><b>%{text}</b>'
    ))
    
    # add the OLS trendline to the figure
    fig.add_trace(go.Scatter(
        x=df[selected_indicator], 
        y=y_pred,
        mode='lines', 
        name='OLS Trendline',
        line=dict(color='darkslategrey')
    ))

    fig.update_layout(
        title=f"{selected_indicator} vs. Perceived Health Status",
        xaxis_title=selected_indicator,
        yaxis_title="Percentage of Fair/Poor Health",
        hovermode='closest'
    )

    return fig

# generate a choropleth map with clusters
def generate_cluster_map(df, geojson):
  
    df = df.copy()
    df['county'] = df['county'].str.title() 
    df['cluster'] = df['cluster'].astype('category')
    fig = px.choropleth_mapbox(df, 
                               geojson=geojson, 
                               color='cluster',
                               locations='county', 
                               featureidkey="properties.COUNTY_NAME",
                               labels={'cluster':'Cluster'},
                               title='County Clusters',
                               mapbox_style="carto-positron",
                               zoom=5.5, 
                               center = {"lat": 36.7783, "lon": -119.4179},  
                               opacity=0.5,
                               hover_name='county',
                               color_discrete_sequence=COLORS
                              )
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

    return fig 

# updates the cluster map figure based on the selected clusters from the checklist
@app.callback(
    Output('cluster-map', 'figure'),
    [Input('cluster-checklist', 'value')]
)
def update_cluster_map(cluster_values):
    # convert the selected clusters to integers and filter the data to include only these clusters
    clusters = [int(cluster) for cluster in cluster_values]
    filtered_df = df[df['cluster'].isin(clusters)]
    
    # generate and return a cluster map using the filtered data
    return generate_cluster_map(filtered_df, counties)

# generate a box plot for a selected feature across different clusters
def generate_box_plot(df, selected_feature):
    # create a box plot with clusters on the x-axis and the selected feature on the y-axis
    fig = px.box(df, x="cluster", y=selected_feature, color="cluster",
                 labels={
                     selected_feature: selected_feature,
                     "cluster": "Cluster"
                 },
                 title=f"Comparison of {selected_feature} in Each Cluster",
                 color_discrete_sequence=COLORS)

    return fig

# updates the box plot figure based on the selected feature from the dropdown
@app.callback(
    Output('cluster-comparison-plot', 'figure'),
    [Input('features-dpdn', 'value')]
)
def update_cluster_comparison_plot(selected_feature):
    # generate and return a box plot using the selected feature
    return generate_box_plot(df, selected_feature)

# run the app
if __name__ == "__main__":
    app.run_server(debug=True)