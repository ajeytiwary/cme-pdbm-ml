## sys loads
import os
import pathlib
from click import style
# import sys
# from click import style
# from nbformat import from_dict
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import mpld3
## import sklearn aux
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

## load regression algos
from sklearn.ensemble import HistGradientBoostingRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor, RandomForestRegressor 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.multioutput import MultiOutputRegressor

## Dash imports 
import dash
from dash import dcc
from dash import html
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State, ClientsideFunction
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import ThemeChangerAIO, template_from_url
from dash import dash_table as dt
import dash_reusable_components as drc
import dash_gif_component as gif
import dash_player
# import seaborn as sns

import tab_explore, tab_experiment
## plotly imports
import plotly.io as pio
import plotly.express as px
from plotly.subplots import make_subplots
import datetime as dtt

# import custom dataloaders
from controls import HALO_STATUS
from load_dfs import load_dataframes

from tab_experiment import header1, dropdown_dataset




# df_cme1,df_cme2,df_cme3,df_cme4  = load_dataframes()
df_cme4  = load_dataframes()
# print(df_cme4.name)

# names = []
# for i in [df_cme1,df_cme2,df_cme3,df_cme4]:
#     print(i.name)

# df_names = []
# for i in [df_cme1,df_cme2,df_cme3,df_cme4]:
#     df_names.append(i.name)

# columns = []
# for i in range(len(df_cme1.columns)): 
#     columns.append(df_cme1.columns[i][0])

# df_cme1.columns = columns


# print(df_names)
cols_test = df_cme4.columns.values
column_names = list(sum(cols_test, ()))[::2]
df_cme4.columns = column_names

# print(df_cme4.columns.values)

def clean_df(df):
    
    df = df.drop(['CME_num','LASCO_Start','Start_Date', 'Arrival_Date', 'Transit_time_err', 'LASCO_Date',  'Analyitic_w',
    'Analyitic_gamma', 'filename', 'v_r_err', 'source_err', 'Bz', 'DST',
    'PE_duration', ], axis =1)
    return df
df_cme4 = clean_df(df_cme4)

# print( df_cme4.head())
Halo_options = df_cme4['LASCO_halo'].unique()






def plot_feature_importance(importance,names,model_type):

    #Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    #Create a DataFrame using a Dictionary
    data={'feature_names':feature_names,'feature_importance':feature_importance}
    fi_df = pd.DataFrame(data)

    #Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)
    trace1 = {
    "uid": "bc205455-f2a3-4d3f-abf3-c40c95a88a07", 
    "type": "bar", 
    "x": fi_df['feature_importance'].to_list(), 
    "y": fi_df['feature_names'].to_list(), 
    "orientation": "h"
    }
    data = go.Data([trace1])
    layout = {
    "title": {"text": "Feature Importance"}, 
    "width": 800, 
    "xaxis": {"title": {"text": "Feature Importance"}}, 
    "yaxis": {
        "title": {
        "font": {"size": 16}, 
        "text": "features"
        }, 
        "tickfont": {"size": 10}
    }, 
    "height": 800, 
    "margin": {"l": 200}, 
    "autosize": False
    }
    fig = go.Figure(data=data, layout=layout)
    return fig



# def error_plot(df):
#     model = LinearRegression()
#     model.fit(X_train, y_train)
#     df['prediction'] = model.predict(X)

#     fig = px.scatter(
#         df, x='petal_width', y='prediction',
#         marginal_x='histogram', marginal_y='histogram',
#         color='split', trendline='ols'
#     )
#     fig.update_traces(histnorm='probability', selector={'type':'histogram'})
#     fig.add_shape(
#         type="line", line=dict(dash='dash'),
#         x0=y.min(), y0=y.min(),
#         x1=y.max(), y1=y.max()
#     )
#     return fig




# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("data").resolve()


# available_indicators = df_cme4.columns.unique()

# available_indicators = []
# for i in df_cme4.columns.unique() :
#    str(available_indicators.append({'label':i,'value':(i)}))

# list_availabe_indicators = df_cme4.columns.unique()

# external_scripts = [
#     'https://www.google-analytics.com/analytics.js',
#     {'src': 'https://cdn.polyfill.io/v2/polyfill.min.js'},
#     {
#         'src': 'https://cdnjs.cloudflare.com/ajax/libs/lodash.js/4.17.10/lodash.core.js',
#         'integrity': 'sha256-Qqd/EfdABZUcAxjOkMi8eGEivtdTkh3b65xCZL4qAQA=',
#         'crossorigin': 'anonymous'
#     }
# ]

# # external CSS stylesheets
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
#     {
#         'href': 'https://stackpath.bootstrapcdn.com/bootstrap/4.1.3/css/bootstrap.min.css',
#         'rel': 'stylesheet',
#         'integrity': 'sha384-MCw98/SFnGE8fJT3GXwEOngsV7Zt27NXFoaoApmYm81iuXoPkFOJwJ8ERdknLPMO',
#         'crossorigin': 'anonymous'
#     }
# ]


# from this file, like this: 
text_markdown = "\t"
with open('README.md') as this_file:
    for a in this_file.read():
        if "\n" in a:
            text_markdown += "\n \t"
        else:
            text_markdown += a



# print(text_markdown)





app = dash.Dash( __name__, meta_tags=[{'name': 'viewport', 'content': 'width=device-width, '
                       'initial-scale=1.0, maximum-scale=1.2, minimum-scale=0.5,'}], external_stylesheets=[dbc.themes.BOOTSTRAP])
    # {"name": "viewport", "content": "width=device-width",  'initial-scale=1.0, maximum-scale=1.2, minimum-scale=0.5,'}])
                # external_scripts=external_scripts,
                # external_stylesheets=external_stylesheets
                # )
app.config.suppress_callback_exceptions = True
app.title = "P-DBM ML Dashboard"
server = app.server


# Custom Script for Heroku
if 'DYNO' in os.environ:
    app.scripts.append_script({
        'external_url': 'https://cdn.rawgit.com/chriddyp/ca0d8f02a1659981a0ea7f013a378bbd/raw/e79f3f789517deec58f41251f7dbb6bee72c44ab/plotly_ga.js'
    })





def get_cols(df):
    dt_col_param = []

    for col in df.columns:
        dt_col_param.append({"name": str(col), "id": str(col)})
    return dt_col_param

def Dat_table(df):
    dt_col_param = get_cols(df)
    return dt.DataTable(
        id='CME_table',
        columns=dt_col_param,
        data=df.to_dict('records'),
        editable = True
    )


data_table = Dat_table(df_cme4)


tabs_styles = {
    'height': '44px',
    'backgroundColor': 'red',
}
tab_style = {
    'borderBottom': '2px solid #d6d6d6',
    'padding': '6px',
    'fontWeight': 'bold'
}

tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#119DFF',
    'color': 'white',
    'padding': '6px'
}

explore_tab = dcc.Tab(label="Explore dataset", value='explore',id="explore",style=tab_style, selected_style=tab_selected_style)

ml_tab = dcc.Tab(label="Gradient boosting experiment", value='experiment', id="experiment",style=tab_style, selected_style=tab_selected_style)


gif_player = gif.GifPlayer(gif='https://www.esa.int/var/esa/storage/images/esa_multimedia/images/2020/10/new_view_of_2012_solar_activity_gif/22288949-2-eng-GB/New_view_of_2012_solar_activity_gif_pillars.gif',#'data/cme_lasco.gif',
                                still= 'https://cdn.sci.esa.int/documents/33795/35391/1567216944476-C2_EIT_410.jpg',
                                autoplay= True,
                                )

title_header = html.Div(
                            [
                                html.H1(
                                    "CME-learn: an intearctive dashboard for P-DBM CME database to run ML Experiments",
                                    style={"margin-bottom": "0px", },
                                ),
                                # dcc.Link(html.H5(
                                #     "Napolitano et al. 2021", style={"margin-top": "0px"}
                                # ), href= 'https://zenodo.org/record/5517629', target='_blank'),
                                dcc.Link(html.H4('Dashboard by Ajay Tiwari'), 
                                href = 'mailto: ajaynld13@gmail.com', target='_blank')   #mailto: abc@example.com
                            ], style ={'text-align': 'center', 'margin-top':50},
                        )



escape_image = html.Div(
                    [
                        html.A(
                            html.Img(
                                src='https://projectescape.eu/sites/default/files/logo-Escape_0.png',
                                style={
                                    'height' : 150,
                                    'width' : 250,
                                    'float' : 'right',
                                    # 'position' : 'relative',
                                    'padding-top' : 0,
                                    'padding-right' : 0, 
                                    'margin-bottom' : 0,
                                    'margin-top': -90,
                                })
                            )
                    ],
                    className="one-third column",
                    id="button",
                    )



## Add raw githb ( check url property)


## Add google drive link and download url

readme_button = html.Div([dbc.Button("Readme", id="open-centered"),
                dbc.Modal(
                    [
                        dbc.ModalHeader(dbc.ModalTitle("Guide!"), close_button=True),
                        dbc.ModalBody(dcc.Markdown(text_markdown), style={"width": "100%"}),
                        dbc.ModalFooter(dbc.Button(
                                                    "Close",
                                                    id="close-centered",
                                                    className="ms-auto",
                                                    n_clicks=0,
                                                ),
                                            ),
                    ],
                    id="modal-centered",
                    is_open=False,
                    scrollable = True,
                    centered=True,
                ),],className="p-5", style = {'float': 'right', 'margin-top': -40} )

paper_button = dbc.Button(
                        id="Paper",
                        children=dcc.Link(html.H5(
                                    "Napolitano et al. 2021", style={"margin-top": "0px"}
                                ), href= 'https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2021SW002925', 
                                          target='_blank'),
                        n_clicks=0,
                        style={
                            # 'borderColor': 'black',
                            'color': 'black',
                            'backgroundColor': ' #97f1eb',
                            'position':'center',
                            'text-align':'center'},
                            
                                
                            )

banner_description = html.Div(
            id='intro-banner',
            className='intro-banner',
            style={'color': 'black',
                   'backgroundColor': '#eef197'},
            children=html.Div(
                className='intro-banner-content',
                children=[
                    html.P(children="A CME machine learning playground to explore and train machine learning models in this application. "
                           "The database used here is descriped in the paper Napolitano et. al. 2021. The description of the model can be found at",
                              
                           className='intro-banner-text'),
                        dbc.Row([dbc.Col(paper_button),
                        dbc.Col(readme_button)],),
                    
                        ],
                    ),
                )
                
                
# colors defined here 
bkg_color = {'dark': '#23262e', 'light': '#f6f6f7'}
grid_color = {'dark': '#53555B', 'light': '#969696'}
text_color = {'dark': '#95969A', 'light': '#595959'}
card_color = {'dark': '#2D3038', 'light': '#FFFFFF'}
accent_color = {'dark': '#FFD15F', 'light': '#ff9827'}


app.layout = dbc.Container([
    html.Div([
        html.Hr(),
            html.Div([gif_player],
                 style={'height' : 200,
                        'width' : 200,
                        'float' : 'left',
                        'position' : 'relative',
                        'padding-top' : 10,
                        'padding-right' : 10, 
                        },
                    className="one-third column",
                ), 
        
            html.Div([title_header],style = {'margin-left': '10%', 'position':'relative', 'width':'80%'},
                    className="one-half column",
                    id="title",),
            escape_image,       
        ], style = {'display':'inline-block','width': '100%' , 'margin-bottom':30}, className = 'row flex-display'),
            # header1,
        html.Hr(),
        
          
            html.Div(className='four columns', children=dropdown_dataset, style={'margin-top':30},),
            banner_description,
            html.Hr(style={'margin-top':30}),
        
        dcc.Tabs( value = 'explore', children= [
                explore_tab, 
                ml_tab,
            ],
            style= tabs_styles,
            id = 'tabs',
            className = 'all-tabs-inline',
        ),
        
        html.Hr(style={'margin-bottom':30}),
        # html.Hr(),
        html.Div(id="tab-content"),
    ],
    className='pretty container', fluid = True)
@app.callback(
        Output("modal-centered", "is_open"),
        [Input("open-centered", "n_clicks"), Input("close-centered", "n_clicks")],
        [State("modal-centered", "is_open")],
    )
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


@app.callback(
    Output("tab-content", "children"),
    [Input("tabs", "value")],
)
def render_tab_content(value):
    """
    This callback takes the 'active_tab' property as input, as well as the
    stored graphs, and renders the tab content depending on what the value of
    'active_tab' is.
    """
    if value is not None:
        if value == "explore":
            return tab_explore.layout
        
        
        elif value == "experiment":
            return tab_experiment.layout
            
    return "Please Select a Tab"


def update_graph(dataset, xaxis_column_name, yaxis_column_name,
                 xaxis_type, yaxis_type,
                 year_slider):
    # df = dataset
    dff = filter_dataframe(dataset, year_slider)

    fig = px.scatter(x=dff[xaxis_column_name],
                     y=dff[yaxis_column_name], color = dff['LASCO_halo'])
                    #  hover_name=dff[dff['Indicator Name'] == yaxis_column_name]['Country Name'])

    fig.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 0}, hovermode='closest')

    fig.update_xaxes(title=xaxis_column_name,
                     type='linear' if xaxis_type == 'Linear' else 'log')

    fig.update_yaxes(title=yaxis_column_name,
                     type='linear' if yaxis_type == 'Linear' else 'log')

    return fig
        

def filter_dataframe( dataset,year_slider):
    df = make_dataset(dataset)
    Halo_options = df['LASCO_halo'].unique()
    # print(df.columns)
    # print(df["LASCO_halo"])
    dff = df[df["LASCO_halo"].isin(Halo_options)
    & (pd.to_datetime(df["LASCO_times"]) > dtt.datetime(year_slider[0], 1, 1))
    & (pd.to_datetime(df["LASCO_times"]) < dtt.datetime(year_slider[1], 1, 1))
    ]
    return dff



@app.callback(
    Output("main_graph", "figure"),
    #  Output('main_graph', 'relayoutData'),
    [Input('dropdown-dataset', 'value'),
     Input("year_slider", "value")],
    )
def make_main_figure(dataset,year_slider):
    df = make_dataset(dataset)
    dff = filter_dataframe(dataset,year_slider)
    figure = go.Figure(px.histogram(dff["LASCO_times"] ,color = dff['LASCO_halo']))
    figure.update_layout(title="Distribution of CMEs in the database", xaxis_title="Date", yaxis_title="Frequency")
    return figure


@app.callback(
    Output("TransTime_v_arr_V", "figure"), 
    [Input('dropdown-dataset', 'value'),
    Input('xaxis-column', 'value'),
    Input('yaxis-column', 'value'),
    Input('xaxis-type', 'value'),
    Input('yaxis-type', 'value'),
    Input('year_slider', 'value')
    ],
)
def update_graph(dataset,xaxis_column_name, yaxis_column_name,
                 xaxis_type, yaxis_type,
                 year_slider):
    df = make_dataset(dataset)

    dff = filter_dataframe(dataset,year_slider)


    fig = px.scatter(dff, x=dff[xaxis_column_name],
                     y=dff[yaxis_column_name], color = dff['LASCO_halo'],
                     marginal_x='histogram', marginal_y='histogram' )
                    #  hover_name=dff[dff['Indicator Name'] == yaxis_column_name]['Country Name'])
    fig.update_traces(histnorm='probability', selector={'type':'histogram'})

    # fig.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 0}, hovermode='closest')

    fig.update_xaxes(title=xaxis_column_name,
                     type='linear' if xaxis_type == 'Linear' else 'log')

    fig.update_yaxes(title=yaxis_column_name,
                     type='linear' if yaxis_type == 'Linear' else 'log')
    fig.update_layout(title_text=xaxis_column_name + ' vs ' + yaxis_column_name, title_x=0.45, 
                      hovermode='closest', title_font_color = 'black', margin=dict(l=20, r=20, t=30, b=20), )
    return fig



@app.callback(
    Output("TransTime_v_Acc", "figure"),
    [Input('dropdown-dataset', 'value'),
    Input('xaxis-column1', 'value'),
    Input('yaxis-column1', 'value'),
    Input('xaxis-type1', 'value'),
    Input('yaxis-type1', 'value'),
    Input('year_slider', 'value')
    ],
)
def make_TTvACC_figure(dataset,xaxis_column_name, yaxis_column_name,
                 xaxis_type, yaxis_type,
                 year_slider):
    df = make_dataset(dataset)
    dff = filter_dataframe(dataset,year_slider)


    fig = px.scatter(dff, x=dff[xaxis_column_name],
                     y=dff[yaxis_column_name], color = dff['LASCO_halo'],
                     marginal_x='histogram', marginal_y='histogram')
                    #  hover_name=dff[dff['Indicator Name'] == yaxis_column_name]['Country Name'])
    fig.update_traces(histnorm='probability', selector={'type':'histogram'})
    # fig.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 0}, hovermode='closest')

    fig.update_xaxes(title=xaxis_column_name,
                     type='linear' if xaxis_type == 'Linear' else 'log')

    fig.update_yaxes(title=yaxis_column_name,
                     type='linear' if yaxis_type == 'Linear' else 'log')
    fig.update_layout(title_text=xaxis_column_name + ' vs ' + yaxis_column_name, title_x=0.45, 
                      hovermode='closest', title_font_color = 'black', margin=dict(l=20, r=20, t=30, b=20),  )

    return fig


@app.callback(
    Output("corr_plot", "figure"), #,'srcDoc'
    [Input('dropdown-dataset', 'value'),
    Input('year_slider', 'value')],
    )
def corr_plot(dataset, year_slider):
   
    df = make_dataset(dataset)
    dff = filter_dataframe(dataset,year_slider)
    corr = dff.corr()

   
    mask = np.zeros_like(corr, dtype = bool)
    mask[np.triu_indices_from(mask)] = True
    corr1=corr.mask(mask)    
    N =len(corr1.columns)
    labels = corr1.columns.values    
    X = [labels[k] for k in range(N)]    
    hovertext = [[f'corr({X[i]}, {X[j]})= {corr.to_numpy()[i][j]:.2f}' if i>j else None for j in range(N)] for i in range(N)]
 
    sns_colorscale = [[0.0, '#3f7f93'], #cmap = sns.diverging_palette(220, 10, as_cmap = True)
    [0.071, '#5890a1'],
    [0.143, '#72a1b0'],
    [0.214, '#8cb3bf'],
    [0.286, '#a7c5cf'],
    [0.357, '#c0d6dd'],
    [0.429, '#dae8ec'],
    [0.5, '#f2f2f2'],
    [0.571, '#f7d7d9'],
    [0.643, '#f2bcc0'],
    [0.714, '#eda3a9'],
    [0.786, '#e8888f'],
    [0.857, '#e36e76'],
    [0.929, '#de535e'],
    [1.0, '#d93a46']]
    heat = go.Heatmap(z=corr1,
                  x=X,
                  y=X,
                  xgap=1, ygap=1,
                  colorscale=sns_colorscale,
                  colorbar_thickness=20,
                  colorbar_ticklen=3,
                  hovertext =hovertext,
                  hoverinfo='text'
                   )
    title = 'Correlation Matrix'               

    layout = go.Layout(title_text=title, title_x=0.5, 
                   width=600, height=600,
                   xaxis_showgrid=False,
                   yaxis_showgrid=False,
                   yaxis_autorange='reversed',
                   plot_bgcolor='rgba(0,0,0,0)',)
   
    fig=go.Figure(data=[heat], layout=layout)  
    return fig
    # fig=go.Figure(corr) 
    
@app.callback(
Output("cme-table", "children"), #,'srcDoc'
[Input('dropdown-dataset', 'value'),
# Input('xaxis-column1', 'value'),
# Input('yaxis-column1', 'value'),
# Input('xaxis-type1', 'value'),
# Input('yaxis-type1', 'value'),
Input('cme_select', 'value')],
    )
def cme_params(dataset, cme_select): 
    df = make_dataset(dataset)
    dict_cme = df.iloc[int(cme_select)].to_dict()
    # dict_cme = 

    # print(type(dict_cme))
    keys = ['CME_characteristis', 'Value']
    col1 = list(dict_cme.keys())
    col2 = list(dict_cme.values())
    cme_dict = {keys[0]: col1, keys[1]:col2} 
    df_cmeprop = pd.DataFrame.from_dict(cme_dict)
    cme_tab = dt.DataTable(                        
                        columns=[{'name': i, 'id': i} for i in keys],
                        data=df_cmeprop.to_dict('records'),
                        fixed_rows={'headers': True},
                        style_cell={'textAlign':'left', 'minWidth': 95, 'width': 95},
                        style_header=dict(backgroundColor="paleturquoise"),
                        style_data=dict(backgroundColor="lavender"),
                        style_table={'height': 600, 'width':600, 'marginTop' : 30},
                        tooltip_delay=0,
                        tooltip_duration=None
                        ),
    return cme_tab
    
    
    #make plot using mpl3D
    
    # mask = np.zeros_like(corr, dtype=bool)
    # mask[np.triu_indices_from(mask)] = True

    # # Want diagonal elements as well
    # mask[np.diag_indices_from(mask)] = False

    # # Set up the matplotlib figure
    # f, ax = plt.subplots(figsize=(11, 9))

    # # Generate a custom diverging colormap
    # cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # # Draw the heatmap with the mask and correct aspect ratio
    # sns_plot = sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
    #         square=True, linewidths=.5, cbar_kws={"shrink": .5},ax = ax)
    # # save to file
    # fig = sns_plot.get_figure() 
    # html_matplot = mpld3.fig_to_html(fig)
    # return html_matplot
    

def run_model(model,X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred_range = model.predict(X_test)
    test_score = model.score(X_test, y_test)
    test_error = mean_squared_error(y_test, model.predict(X_test))    
    
    try:
        importance = model.feature_importances_
    except:
        importance = model.estimators_[0].feature_importances_ 
    
    return model.fit(X_train, y_train), importance





# def make_dataset(name):
    
#     df_list = [df_cme1,df_cme2,df_cme3,df_cme4]
#     ['Earth_directed_cmes', 'SOHO_cmes', 'HELCATS_cmes', 'pdbm_cme']
#     for name in df_names:
#         if name == 'Earth_directed_cmes':
#             df= df_list[0]
#             df = df.round(2)       
#         elif name == 'SOHO_cmes':
#             df= df_list[1]
#             df = df.round(2) 
#         elif name == 'HELCATS_cmes':
#             df= df_list[2]
#             df = df.round(2) 
#         elif name == 'pdbm_cme':
#             df= df_list[3]
#             df = df.round(2)
#             col='LASCO_halo'
#             df[col] = pd.Categorical(df[col], categories=df[col].unique()).codes 
            
#     return df

def make_dataset(name):
    
    # df_list = ['df_cme1','df_cme2','df_cme3',df_cme4]
    # df_names = ['Earth_directed_cmes', 'SOHO_cmes', 'HELCATS_cmes', 'pdbm_cme']
    # for name in df_names:
    #     if name == 'Earth_directed_cmes':
    #         df= df_list[0]
    #         df = df.round(2)       
    #     elif name == 'SOHO_cmes':
    #         df= df_list[1]
    #         df = df.round(2) 
    #     elif name == 'HELCATS_cmes':
    #         df= df_list[2]
    #         df = df.round(2) 
    if name == 'pdbm_cme':
        df= df_cme4
        df = df.round(2)
        col='LASCO_halo'
        df[col] = pd.Categorical(df[col], categories=df[col].unique()).codes 
            
    return df

def feature_df(df, feature_columns):
    X = df[feature_columns]
    return X

def target_df(df, target_columns):
    y = df[target_columns]
    return y

def str_params(params):
    for keys in params:
            params[keys] = str(params[keys])
    return params



@app.callback(Output('slider-polynomial-degree', 'disabled'),
              [Input('dropdown-select-model', 'value')])
def disable_slider_alpha(model):
    return model not in ['lasso', 'ridge', 'elastic_net']


@app.callback(Output('slider-alpha', 'disabled'),
              [Input('dropdown-select-model', 'value')])
def disable_slider_alpha(model):
    return model not in ['lasso', 'ridge', 'elastic_net']


@app.callback(Output('slider-l1-l2-ratio', 'disabled'),
              [Input('dropdown-select-model', 'value')])
def disable_dropdown_select_model(model):
    return model not in ['elastic_net']



@app.callback([Output('graph-regression-display', 'figure'),
               Output('graph-error-display', 'figure'), 
               Output('params-table', 'children' )],
              [Input('dropdown-dataset', 'value'),
               Input('dropdown-select-model', 'value'),
               Input('feature_columns', 'value'),               
               Input('target_columns', 'value'),             
               Input('slider-polynomial-degree', 'value'),          
               Input('slider-alpha', 'value'),
               Input('slider-l1-l2-ratio', 'value')])
def update_graph(dataset, model_name, feature_columns, target_columns, degree, alpha_power,  l2_ratio):
    # print('tesing here')
    # print(dataset, model_name, feature_columns, target_columns, degree, alpha_power,  l2_ratio )
    df=  make_dataset(dataset)
    # print(df.head())
    # print(feature_columns, target_columns)
    X = feature_df(df, feature_columns)
    # print(X)

    y = target_df(df, target_columns)
    
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.25, random_state=123)

    X_range = np.linspace(X.min() - 0.5, X.max() + 0.5, 300).reshape(-1, 1)

    degree =1
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.fit_transform(X_test)
    poly_range = poly.fit_transform(X_range)
    # print(len(target_columns))
    # Select model
    alpha = 10 ** alpha_power
    if model_name == 'lasso':
        model = Lasso(alpha=alpha, normalize=True)
        
    elif model_name == 'ridge':
        model = Ridge(alpha=alpha, normalize=True)
        
    elif model_name == 'elastic_net':
        model = ElasticNet(alpha=alpha, l1_ratio=1 - l2_ratio, normalize=True)
        
    elif (model_name == 'gradient_boost') and (len(target_columns) ==1):
        model = GradientBoostingRegressor()
        model.fit(X_train, y_train)
        y_pred_range = model.predict(X_test)
        test_score = model.score(X_test, y_test)
        test_error = mean_squared_error(y_test, model.predict(X_test))        
        importance = model.feature_importances_
        params = model.get_params()
        params = str_params(params)
        
    elif (model_name == 'gradient_boost') and (len(target_columns) >1):
        model = MultiOutputRegressor(GradientBoostingRegressor())
        model.fit(X_train, y_train)
        y_pred_range = model.predict(X_test)
        test_score = model.score(X_test, y_test)
        test_error = mean_squared_error(y_test, model.predict(X_test))
        importance = model.estimators_[0].feature_importances_   
        params = model.get_params()
        params = str_params(params)
        
    elif (model_name == 'xgboost') and (len(target_columns) ==1):        
        model = XGBRegressor()
        model_fit, importance = run_model(model,X_train, y_train,X_test, y_test )
        params = model.get_xgb_params()
        params = str_params(params)
        
    elif (model_name == 'xgboost') and (len(target_columns) >1):
        model =   MultiOutputRegressor(XGBRegressor())  
        model_fit, importance = run_model(model,X_train, y_train,X_test, y_test )
        params = model.get_params()
        params = str_params(params)
    
    elif (model_name == 'lightgbm') and (len(target_columns) ==1):        
        model = LGBMRegressor()
        model_fit, importance = run_model(model,X_train, y_train,X_test, y_test )
        params = model.get_params()
        params = str_params(params)
        
    elif (model_name == 'lightgbm') and (len(target_columns) >1):
        model =   MultiOutputRegressor(LGBMRegressor())  
        model_fit, importance = run_model(model,X_train, y_train,X_test, y_test )    
        params = model.get_params()
        params = str_params(params)
        
    elif (model_name == 'catboost') and (len(target_columns) ==1):        
        model = CatBoostRegressor()
        model_fit, importance = run_model(model,X_train, y_train,X_test, y_test )
        params = model.get_all_params()
        params = str_params(params)
        
    elif (model_name == 'catboost') and (len(target_columns) >1):
        model =   MultiOutputRegressor(CatBoostRegressor())  
        model_fit, importance = run_model(model,X_train, y_train,X_test, y_test )  
        params = model.get_params()  
        params = str_params(params)
    # elif (model_name == 'hist_gradient_boost') and (len(target_columns) ==1):        
    #     model = HistGradientBoostingRegressor()
    #     model_fit, importance = run_model(model,X_train, y_train,X_test, y_test )
    #     params = model.get_all_params()
    #     params = str_params(params)
        
    # elif (model_name == 'hist_gradient_boost') and (len(target_columns) >1):
    #     model =   MultiOutputRegressor(HistGradientBoostingRegressor())  
    #     model_fit, importance = run_model(model,X_train, y_train,X_test, y_test )  
    #     params = model.get_params()  
    #     params = str_params(params)
 
    else:
        model = LinearRegression(normalize=True)
                                                            
    # print(model.get_params())  
    # model.get_all_params                                                      
    names = feature_columns
    # print(params.items())
    # df_params = pd.DataFrame.from_dict(params, orient = 'index')
    # df_params = df_params.T
    # print(df_test)
    
    # for k, v in params.items():
    #     value = v
    #     print(k,value)
    keys = ['Model_parameters', 'Value']
    # param_dict = {keys[0]:list(params.keys()),keys[1]:list(params.values())}
    # ty  = param_dict.values()
    col1 = list(params.keys())
    col2 = list(params.values())
    
    # print(type(col1))
    
    params_dict =  {keys[0]: col1, keys[1]:col2}                 #dict(zip(col1,col2))
    # print(params_dict.keys())
    
    # print(params, params_dict)
    df_params = pd.DataFrame.from_dict(params_dict)
    # print(df_params.T.head())
    param_table = dt.DataTable(
                        
                        columns=[{'name': i, 'id': i} for i in keys],
                        data=df_params.to_dict('records'),
                        fixed_rows={'headers': True},
                        style_cell={'textAlign':'left', 'minWidth': 15, 'maxWidth': 30, 'height': 'auto', 'lineHeight':'15px','overflowX': 'hidden', 'textOverflow': 'ellipsis'},
                        style_header=dict(backgroundColor="paleturquoise"),
                        style_data=dict(backgroundColor="lavender", whiteSpace = 'normal',height = 'auto'),
                        style_table={'height': 300, 'width':450, 'marginTop' : 20, 'overflowX': 'scroll'},
                        css=[{
                                'selector': '.dash-cell div.dash-cell-value',
                                'rule': 'display: inline; white-space: inherit; overflow: inherit; text-overflow: inherit;'
                            }],
                        ),
         
    # pd.
    # df_params = pd.DataFrame.from_dict(params )
    # print(df.params.head())
    #Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    #Create a DataFrame using a Dictionary
    data={'feature_names':feature_names,'feature_importance':feature_importance}
    fi_df = pd.DataFrame(data)

    #Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=True,inplace=True)
    
    fig = px.bar(x=fi_df['feature_importance'].to_list(), y =fi_df['feature_names'].to_list(), orientation='h', title="Feature Importance") #, error_x=std add error
    fig.update_layout(title_text = "Feature Importance for prediction", title_x = 0.45)
    fig.update_yaxes(title_text = 'Features')
    fig.update_xaxes(title_text = 'Feature importance')
    # fig2 = go.Figure([go.Bar(x=fi_df['feature_importance'].to_list(), y =fi_df['feature_names'].to_list(), orientation='h' )]) #, error_x=std add error
    # y_test.shape()
    y_pred_test = model.predict(X_test)
    # print(y_test[target_columns].to_numpy().flatten())
    if len(target_columns) ==1:
        fig2 = px.scatter(x = y_test[target_columns].to_numpy().flatten(), y= y_pred_test ) #,labels={'x': '(ground truth)', 'y':  '(prediction)'}, 
                        #   title = 'Comparing the prediction and ground truth', title_x = 0.45)
        fig2.add_shape(
        type="line", line=dict(dash='dash'),
        x0=y_pred_test.min(), y0=y_pred_test.min(),
        x1=y_pred_test.max(), y1=y_pred_test.max()
                )
        fig2.update_yaxes(title_text = target_columns[0] + " (ground truth)")
        fig2.update_xaxes(title_text = target_columns[0] + " (prediction)")
        fig2.update_layout(title_text = "Comparing the prediction and ground truth", title_x = 0.45)
    else:
        fig2 = make_subplots(rows=1, cols=2, horizontal_spacing = 0.15)        
        fig2.add_trace(go.Scatter(x = y_test[target_columns[0]].to_numpy().flatten().tolist(), 
                                 y= y_pred_test[:,0].tolist(), mode="markers", name = target_columns[0]),
                       row=1, col=1
                       )
                
        fig2.add_trace(go.Scatter(x = y_test[target_columns[1]].to_numpy().flatten().tolist(), 
                                 y= y_pred_test[:,1].tolist(), mode="markers", name = target_columns[1]),
                       row=1, col=2 
                       )
        fig2.update_yaxes(title_text = target_columns[0] + " (ground truth)", row=1, col=1)
        fig2.update_xaxes(title_text = target_columns[0] + " (prediction)", row=1, col=1)
        fig2.update_yaxes(title_text = target_columns[1] + " (ground truth)", row=1, col=2)
        fig2.update_xaxes(title_text = target_columns[1] + " (ground truth)", row=1, col=2)
        fig2.update_layout(title_text = "Comparing the prediction and ground truth", title_x = 0.45)

    # param_table = dt.DataTable(
                        
    #                     columns=[{"name": i, "id": i} for i in params.keys()],
    #                     data=[params],
    #                     style_table={'width': '300px', 'overflowY': 'auto', 'rotate':'90(deg)'}),
    return fig, fig2, param_table



# Running the server
if __name__ == '__main__':
    app.run_server(threaded=True, debug=True)
