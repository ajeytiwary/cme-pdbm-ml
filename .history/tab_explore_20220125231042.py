
# from multiprocessing.sharedctypes import Value
# from sqlite3 import Row


from dash import dcc
from dash import html

# from dash.dependencies import Input, Output, State
# import dash_bootstrap_components as dbc
# from dash_bootstrap_templates import ThemeChangerAIO, template_from_url
# from dash import dash_table as dt
import dash_reusable_components as drc
# import dash_player


# import plotly.graph_objs as go
# import plotly.io as pio
# import plotly.express as px

import pandas as pd



from controls import HALO_STATUS
# halo_options = ['HH', 'FH', 'PH']
from load_dfs import load_dataframes

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

heading_slider =  html.P(
                            "Filter by CME date (or select range in histogram):",
                            className="control_label",
                        )

slider = dcc.RangeSlider(
                            id="year_slider",
                            marks = {i: '{}'.format(i) for i in range(years.min(), years.max())},
                            min=years.min(),
                            max=years.max(),
                            value=[years.min(), years.max()],
                            className="dcc_control",
                        )

dist_graph = dcc.Graph(id="main_graph",config= {'displaylogo': False})

dropdown_1_x = drc.NamedDropdown(name = 'Select X-axis',
                                id='xaxis-column',
                                options=[{'label': i, 'value': i} for i in available_indicators],
                                value='Arrival_v',
                                clearable = False,
                                # style = {"margin-bottom": "25px"}
                            )

radio_1_x = dcc.RadioItems(
                                id='xaxis-type',
                                options=[{'label': i, 'value': i} for i in ['Linear', 'Log']],
                                value='Linear',
                                labelStyle={'display': 'inline-block'},
                                inputStyle={"margin-top": "10px"},
                                # inputStyle={"margin-right": "5px", "margin-left": "20px"},
                                
                            )
dropdown_1_y = drc.NamedDropdown(name = 'Select Y-axis',
                                id='yaxis-column',
                                options=[{'label': i, 'value': i} for i in available_indicators],
                                value='Transit_time',
                                clearable = False,
                                # style = {"margin-bottom": "25px"}
                            )
radio_1_y = dcc.RadioItems(
                                id='yaxis-type',
                                options=[{'label': i, 'value': i} for i in ['Linear', 'Log']],
                                value='Linear',
                                labelStyle={'display': 'inline-block'},
                                inputStyle={"margin-top": "10px"},
                                )




dropdown_2_x = drc.NamedDropdown(name = 'Select X-axis',
                            id='xaxis-column1',
                            options=[{'label': i, 'value': i} for i in available_indicators],
                            value='Accel.',
                            clearable = False,
                        )
radio_2_x =  dcc.RadioItems(
                            id='xaxis-type1',
                            options=[{'label': i, 'value': i} for i in ['Linear', 'Log']],
                            value='Linear',
                            labelStyle={'display': 'inline-block'},
                            inputStyle={"margin-top": "10px"},
                        
                        )




 
dropdown_2_y = drc.NamedDropdown(name = 'Select Y-axis',
                        id='yaxis-column1',
                        options=[{'label': i, 'value': i} for i in available_indicators],
                        value='Transit_time',
                        clearable = False,
                        # style = {'margin-left' : '55%', 'width' : '90%',"margin-bottom": "25px"}
                        
                    )
radio_2_y = dcc.RadioItems(
                        id='yaxis-type1',
                        options=[{'label': i, 'value': i} for i in ['Linear', 'Log']],
                        value='Linear',
                        labelStyle={'display': 'inline-block'},
                        inputStyle={"margin-top": "10px"},
                        # style = {'margin-left' : '90%', 'width' : '60%'},
                        # inputStyle={"margin-right": "5px", "margin-left": "20px"},
                    )


x_y_plot1 = dcc.Graph(id="TransTime_v_arr_V",config= {'displaylogo': False},)


x_y_plot2 = dcc.Graph(id="TransTime_v_Acc",config= {'displaylogo': False}, )

corr_plotted = dcc.Graph(id="corr_plot",config= {'displaylogo': False},)


dropdown_cme = drc.NamedDropdown(name = 'Select CME',
                                id='cme_select',
                                options=[{'label': i, 'value': i} for i in df.index],
                                value='1',
                                clearable = False,
                                style= {'width':590}
                                )



                # video_player = dash_player.DashPlayer(
                #             id = 'video_player',
                #             url = "https://cdaw.gsfc.nasa.gov/CME_list/nrl_mpg/2013_09/130901_c3.mpg",
                #             # url = 'http://127.0.0.1:8000/video_5.mp4',
                #             # width = 900,
                #             # height = 720,
                #             # controls = True,
                #             # # cursorUpdate = 5000,
                #             # playing = True 
                #             )
                
                

# @app.callback(Output('video-player', 'playing'),
#               [Input('radio-bool-props', 'values')])
# def update_prop_playing(values):
#     return 'playing' in values


# @app.callback(Output('video-player', 'loop'),
#               [Input('radio-bool-props', 'values')])
# def update_prop_loop(values):
#     return 'loop' in values


# @app.callback(Output('video-player', 'controls'),
#               [Input('radio-bool-props', 'values')])
# def update_prop_controls(values):
#     return 'controls' in values


# @app.callback(Output('video-player', 'muted'),
#               [Input('radio-bool-props', 'values')])
# def update_prop_muted(values):
#     return 'muted' in values


# @app.callback(Output('video-player', 'seekTo'),
#               [Input('radio-bool-props', 'values')])
# def update_prop_seekTo(values):
#     if 'seekTo' in values:
#         return 5

cross_validation_slider = drc.NamedSlider(
                    name='Cross validations (number of folds)',
                    id='CV_fold',
                    min=0,
                    max=len(df_cme4),
                    step=1,
                    value=3,
                    # marks={0: 'L1', 1: 'L2'}
                )



# dropdown_cme_num = 'dropdown'

h5_cme_details = 'html adding things: CME details as a cardwith cme_movie and below the details'
# helioviewer and other things




def explore():
     label="Explore dataset", 
     value= 'explore',
     #start main div
     
     layout = html.Div(className="heading", children=[ 
        #  html.Div(id="output-clientside"),
        #div for header
         html.Div([
            #  html.Div([
            #      html.Div([gif_player],
            #      style={'height' : '10%',
            #             'width' : '10%',
            #             'float' : 'left',
            #             # 'position' : 'relative',
            #             'padding-top' : 0,
            #             'padding-right' : 0,
            #             },
            #             className="one-third column",),
            #             # 
            #             ]),
            #       html.Div(
            #                 [title_header],
            #                 style = {'margin-left': '10%'},
            #                 className="one-half column",
            #                 id="title",
            #                 ),
            #                 escape_image,
            #                 id="header",
            #                 className="row flex-display",
                    #     style={"margin-bottom": "50px"},
                    # ),
                    #     ],
                    #         ),
             
                    #         html.Div(
                    #             [title_header],
                    #             style = {'margin-left': '10%'},
                    #             className="one-half column",
                    #             id="title",
                    #         ),
                    #         escape_image
                    #     ],
                    #     id="header",
                    #     className="row flex-display",
                    #     style={"margin-bottom": "50px"},
                    # ),
                    #     ],
                    #         ),
     
             
            html.Div([
                    heading_slider,
                    slider,

                    html.Div([dist_graph],
                            id="countGraphContainer",
                            className="pretty_container",
                        ),
                    ],

                    id="right-column",
                    className="fourteen columns",
                ),
            ],
            className="row flex-display", style={'margin-bottom':'30px'},
        ),
         
        #div header verified 
         
            html.Div([
            html.Spacer(style = {'padding':'15px'}),
            
            html.Div(children  = [
                html.Div(className='pretty_container', children  = [
                            dropdown_1_x,
                            radio_1_x,
                        ], 
                        #  className="pretty_container",
                         style={ 'width':'30%', 
                                'display':'inline-block', 'text-align': 'center', 'verticalAlign' : "middle",'margin-left': '50px',}
                        ),

                html.Div(className='pretty_container', children  = [
                        dropdown_1_y,
                        radio_1_y,
                        ], 
                        # className="pretty_container", 
                        style={'width':'30%',  'display':'inline-block', 
                               'text-align': 'center', 'verticalAlign' : "middle",'margin-left': '50px',}
                        ), 
                html.Div ([x_y_plot1],
                            id="TT_V_acc",
                            className="pretty_container five columns",
                            style = {'margin-top':'90px', 'margin-left':'5px'},
                            # className="pretty_container eight columns",
                            ),
            ], 
                     style = {'width':'45%','display':'inline-block' }, 
                     #side note two +1
                     ),
            
            
            html.Spacer(style = {'padding':'25px'}),
            
            html.Div([
                html.Div(className='pretty_container', children  = [
                        dropdown_2_x,
                        radio_2_x,
                    ], 
                    #   className="pretty_container",
                      style={'width':'30%', 'display':'inline-block', 'text-align': 'center', 'verticalAlign' : "middle",
                             'margin-left': '50px', }
                        
                    #style={'align-items': 'center', 'border': '1px solid black', 'display': 'flex', 'justify-content': 'center'}
                    ),
 
                html.Div(className='pretty_container', children  = [
                        dropdown_2_y,
                        radio_2_y,
                        ],
                        #  className="pretty_container",
                        style={'width':'30%', 'display':'inline-block', 'text-align': 'center', 'verticalAlign' : "middle", 'margin-left': '50px'}
                            ), #], style = {'margin-bottom': '50px'}),

                html.Div([x_y_plot2],
                        id="TT_V_arrV",
                        style = {'margin-top':'90px'},
                        className="pretty_container ",
                                ),
                    ], 
                     style = {'width':'45%','display':'inline-block' }, #style = {'width':'49%'},
                     ),],
                     className = 'row',
                     style = {'display':'block'},
             ),
            html.Div([
                # html.Iframe(srcDoc = 'html_matplot',
                html.Div(corr_plotted,
                        style = {'width':'45%','display':'inline-block' ,'margin-top':'30px'},
                        className="pretty_container",
                                ),
                
                html.Spacer(style = {'padding':'50px'}),
                
                html.Div(className='pretty_container', children  = [
                        dropdown_cme,
                        html.Div(id='cme-table', ),                         
                        ],
                        style = {'width':'45%','display':'inline-block', 'position':'absolute' },
                        
                                ),
            ], style = {'display':'block'})
 


                ])
     
     return layout, label, value


layout, label, value = explore()