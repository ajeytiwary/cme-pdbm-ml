
from dash import dcc
from dash import html
from dash import dash_table as dt
import dash_reusable_components as drc
import dash_player


# from dash.dependencies import Input, Output, State
# import dash_bootstrap_components as dbc
# from dash_bootstrap_templates import ThemeChangerAIO, template_from_url


import pandas as pd
## plotly imports
# import plotly.io as pio
# import plotly.express as px
# import plotly.graph_objs as go


from controls import HALO_STATUS
# halo_options = ['HH', 'FH', 'PH']
from load_dfs import load_dataframes



# df_cme1,df_cme2,df_cme3,df_cme4  = load_dataframes()
df_cme4  = load_dataframes()
# df = df_cme4

# columns = []
# for i in range(len(df.columns)): 
#     columns.append(df.columns[i][0])

# df.columns = columns

# Halo_options = df['LASCO_halo'].unique()


# cme_types= [
#     {"label": str(HALO_STATUS[cme_type]), "value": str(cme_type)}
#     for cme_type in HALO_STATUS
# ]



# # print(cme_types)

# # cme_types = Halo_options

# df["LASCO_Start"] = pd.to_datetime(df["LASCO_Start"])
# # df = df[pd.to_datetime(df["LASCO_Start"]) > dt.datetime(1990, 1, 1)]



# years = pd.DatetimeIndex(df['LASCO_Start']).year.unique()

# df['Year'] = pd.DatetimeIndex(df['LASCO_Start']).year

# df_plot  =  df.drop(labels = [df.columns[0],df.columns[1],df.columns[2],df.columns[3],df.columns[7],df.columns[8],
# df.columns[14],df.columns[17],df.columns[26],df.columns[27],df.columns[28],df.columns[29]], axis =1 )

# available_indicators = df_plot.columns.unique()


df_cme4  = load_dataframes()

df = df_cme4

columns = []
for i in range(len(df.columns)): 
    columns.append(df.columns[i][0])

df.columns = columns

Halo_options = df['LASCO_halo'].unique()


cme_types= [
    {"label": str(HALO_STATUS[cme_type]), "value": str(cme_type)}
    for cme_type in HALO_STATUS
]



# print(cme_types)

# cme_types = Halo_options

df["LASCO_Start"] = pd.to_datetime(df["LASCO_Start"])
# df = df[pd.to_datetime(df["LASCO_Start"]) > dt.datetime(1990, 1, 1)]



years = pd.DatetimeIndex(df['LASCO_Start']).year.unique()

df['Year'] = pd.DatetimeIndex(df['LASCO_Start']).year

# df_plot  =  df.drop(labels = [df.columns[0],df.columns[1],df.columns[2],df.columns[3],df.columns[7],df.columns[8],
# df.columns[14],df.columns[17],df.columns[26],df.columns[27],df.columns[28],df.columns[29]], axis =1 )

def clean_df(df):
    
    df = df.drop(['CME_num','LASCO_Start','Start_Date', 'Arrival_Date', 'Transit_time_err', 'LASCO_Date',  'Analyitic_w',
    'Analyitic_gamma', 'filename', 'v_r_err', 'source_err', 'Bz', 'DST',
    'PE_duration','Year'], axis =1)
    return df
df_cme4 = clean_df(df_cme4)

# print( df_cme4.head())
# Halo_options = df_cme4['LASCO_halo'].unique()

available_indicators = df_cme4.columns.unique()
# print(available_indicators)

target_indicators = available_indicators[0:2]
# print(target_indicators)
feature_indicators = available_indicators[2:-2]



# param_table = dt.DataTable(
#     id='params-table',
#     columns=[{"name": i, "id": i} for i in params.key],
#     data=params,
# )




header1 = html.H2(html.A(
                'CME PDBM Gradient Boosting Experiment',
                href='https://cme-pdbm.herokuapp.com/',
                style={'text-decoration': 'none', 'color': 'inherit'}
            ))


dropdown_dataset = drc.NamedDropdown(
                name='Select Dataset',
                id='dropdown-dataset',
                options=[
                    {'label': 'PDBM Database (Napolitano et. al. 2021)', 'value': 'pdbm_cme'},
                    # {'label': 'CME datbase (Greek)', 'value': 'CME_dat_greece'},
                    # {'label': 'CME datbase (Helcats)', 'value': 'CME_dat_helacts'},
                    # {'label': 'CME datbase (SOHO)', 'value': 'CME_dat_SOHO'},
                    ],
                value='pdbm_cme',
                clearable=False,
                searchable=False
            )



dropdown_models = drc.NamedDropdown(
                name='Select Model',
                id='dropdown-select-model',
                options=[
                    # {'label': 'Linear Regression', 'value': 'linear_reg'},
                    # {'label': 'Lasso', 'value': 'lasso'},
                    # {'label': 'Ridge', 'value': 'ridge'},
                    # {'label': 'Elastic Net', 'value': 'elastic_net'},
                    {'label': 'Gradient Boost', 'value': 'gradient_boost'},
                    {'label': 'XGBoost', 'value': 'xgboost'},
                    {'label': 'LightGBM', 'value': 'lightgbm'},
                    {'label': 'CatBoost', 'value': 'catboost'},
                    # {'label': 'Histogram Gradient Boost', 'value': 'hist_gradient_boost'},
                    
                    # {'label': 'ExtraTrees Regression', 'value': 'extra_trees'},
                    # {'label': 'Random Forest regression', 'value': 'random_forest'},
                    ],
                value='gradient_boost',
                searchable=False,
                clearable=False
            )


dropdown_features = drc.NamedDropdown(name='Input Features ',
                            id='feature_columns',
                            options = [{"label": str(i), "value": i} for i in feature_indicators],
                            multi=True,
                            value=['LASCO_v', 'v_r'],
                            className="dcc_control",
                            clearable = False,
                        )

dropdown_targets = drc.NamedDropdown(name='Target features',
                            id='target_columns',
                            options = [{"label": str(i), "value": i} for i in target_indicators],
                            multi=True,
                            value=['Transit_time'], 
                            className="dcc_control",
                            clearable = False,
                        )


degree_slider = drc.NamedSlider(
                name='Cross Validation (number of folds)',
                id='slider-polynomial-degree',
                min=1,
                max=10,
                step=1,
                value=1,
                marks={i: str(i) for i in range(1, 11)},
            )



alpha_slider = drc.NamedSlider(
                name='Alpha (Regularization Term)',
                id='slider-alpha',
                min=-4,
                max=3,
                value=0,
                marks={i: '{}'.format(10 ** i) for i in range(-4, 4)}
            )


ratio_slider = drc.NamedSlider(
                    name='L1/L2 ratio (Select Elastic Net to enable)',
                    id='slider-l1-l2-ratio',
                    min=0,
                    max=1,
                    step=0.05,
                    value=0.5,
                    marks={0: 'L1', 1: 'L2'}
                )
 

graph_main = dcc.Graph(
                        id='graph-regression-display',
                        className='row',
                        style={'height': 'calc(100vh - 160px)'},
                        config= {'displaylogo': False},
                        # config={'modeBarButtonsToRemove': [
                        #     'pan2d',
                        #     'lasso2d',
                        #     'select2d',
                        #     'autoScale2d',
                        #     'hoverClosestCartesian',
                        #     'hoverCompareCartesian',
                        #     'toggleSpikelines',
                        #     'displaylogo'

                        # ]}
                    )

graph_error = dcc.Graph(
                        id='graph-error-display',
                        className='row',
                        style={'height': 'calc(100vh - 160px)'},
                        config= {'displaylogo': False},
                        # config={'modeBarButtonsToRemove': [
                        #     'pan2d',
                        #     'lasso2d',
                        #     'select2d',
                        #     'displaylogo',
                        #     'autoScale2d',
                        #     'hoverClosestCartesian',
                        #     'hoverCompareCartesian',
                        #     'toggleSpikelines',
                        #     'displaylogo',

                        # ]}
                    )


video_player = dash_player.DashPlayer(
            id = 'video_player',
            url = "https://cdaw.gsfc.nasa.gov/CME_list/nrl_mpg/2013_09/130901_c3.mpg",
            # url = 'http://127.0.0.1:8000/video_5.mp4',
            # width = 900,
            # height = 720,
            # controls = True,
            # # cursorUpdate = 5000,
            # playing = True 
            )


cross_validation_slider = drc.NamedSlider(
                    name='Cross validations (number of folds)',
                    id='CV_fold',
                    min=0,
                    max=len(df_cme4),
                    step=1,
                    value=3,
                    # marks={0: 'L1', 1: 'L2'}
                )

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






def experiment():
     layout = html.Div(className="banner", children=[
         
        # html.Div(className='container scalable', children=[
        #     header1,
        #     # html.A(
        #     #     html.Img(src="https://s3-us-west-1.amazonaws.com/plotly-tutorials/logo/new-branding/dash-logo-by-plotly-stripe-inverted.png"),
        #     #     href='https://plot.ly/products/dash/'
        #     # )
        # ]),
    
        # html.Div(id='body', className='container scalable', children=[
        # html.Div(
        #     className='row',
        #     style={'padding-bottom': '10px'},
        #     children=dcc.Markdown("""
        #     [Click here](https://github.com/plotly/dash-regression) to visit 
        #     the project repo, and learn about how to use the app.
        #     """)
        # ),
        # html.Iframe(
        #     # controls = True,
        #     id = 'movie_player',
        #     #src = "https://www.youtube.com/watch?v=gPtn6hD7o8g",
        #     src = f"https://sdo.gsfc.nasa.gov/assets/img/browse/2019/04/04/20190404_004559_512_HMI171.jpg",
        #     # type="video/mp4",
        #     # autoPlay=True
        # ),
        # html.Div([dash_player.DashPlayer(
        #     id = 'video_player',
        #     url = "http://media.w3.org/2010/05/bunny/movie.mp4",
        #     # https://cdaw.gsfc.nasa.gov/CME_list/nrl_mpg/2013_09/130901_c3.mpeg",
        #     # url = 'http://127.0.0.1:8000/video_5.mp4',
        #     # width = 900,
        #     # height = 720,
        #     controls = True,
        #     # # cursorUpdate = 5000,
        #     # playing = True 
        #     ),
    #     dcc.Checklist(
    #     id='radio-bool-props',
    #     options=[{'label': val.capitalize(), 'value': val} for val in [
    #         'playing',
    #         'loop',
    #         'controls',
    #         'muted',
    #         'seekTo',
    #     ]],
    #     value=['controls'],
    # ),
                 
        # ]),

        # html.Div(id='custom-data-storage', style={'display': 'none'}),

        html.Div([html.Div(className='row',children=[
            # html.Div(className='four columns', children=dropdown_dataset),
            html.Div(className='two columns', children= dropdown_features),
            
            html.Div(className='two columns', children= dropdown_targets),
            
            html.Div(className='two columns', children=dropdown_models),
        ],style={'width': '30%', 'display': 'inline-block'}),
        
        html.Spacer(style = {'padding':'10px'}),
        
        html.Div(className='row', children=[
            
                html.Div(className='two columns', children=degree_slider),

                html.Div(className='two columns', children=alpha_slider),

                html.Div(className='two columns', children=ratio_slider),
                # html.Div(id='params-table'),
                
        ],style={'width': '25%', 'display': 'inline-block', 'position':'absolute'}),
        
        html.Spacer(style = {'padding':'20px'}),
        
        html.Div(className='row', children = [
                html.Div(id='params-table', ),
                              
                ], 
                 style ={'width': '10%', 'display': 'inline-block', 'position':'absolute', 'margin-bottom':'50px', 'margin-left':5, 'margin-top':-10}
                #  style ={'width': '20%', 'display': 'inline-block', 'height': '20%', 'margin-top':'-30px'},
                #  style = {'width': '20%', 'display': 'inline-block', 'marginBottom': 50, 'marginTop': -125}
                 ),
        
        ], style = {'display':'block'}),

        # html.Spacer(style = {'padding':'60px'}),
        
        # html.Div(children = data_table),#Dat_table), #id='CME_table',  className='tableDiv'),
        html.Div(className='three-columns',children=[
            graph_main,            
        ],
                 style = {'width':'45%','display':'inline-block' ,'margin-top':'150px'}, 
                 ),
        html.Spacer(style = {'padding':'60px'}),

        html.Div(className='three-columns',children=[
            graph_error,
        ],
                 style = {'width':'45%','display':'inline-block' ,'margin-top':'150px'}, 
                 ),
    ])
     return layout
     




layout= experiment()