# data
import json
import pandas as pd
import numpy as np

# visualization
from bokeh.models.sources import ColumnDataSource, CDSView
from bokeh.layouts import column, gridplot, row
from bokeh.transform import transform, Jitter, linear_cmap
from bokeh.models import (
                        DataTable, TableColumn, Panel, Tabs,
                        Button, Slider, MultiChoice,Dropdown,RadioButtonGroup,
                        ColorBar, LinearColorMapper,
                        )

from bokeh.palettes import RdYlBu5, Category10, Turbo256, Inferno256
from bokeh.plotting import figure, curdoc, show
from jinja2 import Template

# machine learning
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# for the command line:

# conda activate bokehDashboard
# cd <path to myapp>
# bokeh serve --show mainPart2.py

# use your browser and go to this address to see your dashboard 
# http://localhost:5006/


# ==== util functions ====
def get_second(x):
  return x[1]
def get_rank(data):
    dataID = [(i,x) for i, x in enumerate(data)]
    dataID.sort(key=get_second)
    ranks = [0]* len(data)
    for i in range(len(ranks)):
        ranks[dataID[i][0]] = i
    return ranks
#=============================

# load data
csvPath =  r".\data\unsupervised-clustering-data.csv"
df = pd.read_csv(csvPath)

#===============================================
# ===== Part I: understanding the data with tables, distribution plots and maps =====

# get base stats on each column/varaible of our dataset
df_description = df.describe().T

# transform to bokeh data format 
tableSource = ColumnDataSource(df_description)

# create a interactive table
tableColumns = []
for colName in tableSource.data.keys():
    tableColumns.append(TableColumn(field=colName, title=colName))
data_table = DataTable(source=tableSource, columns=tableColumns, width=600, height=900)

#===============================================
# ==== creating figures and plotting data  =====

# create ColumnDataSource for rank/value + map plot
distData = pd.DataFrame()
distData["value"]= df["strOri_0"]
distData["rank"] = get_rank(df["strOri_0"])
distData["x_coord"] =  df["x_coord"]
distData["y_coord"] = df["y_coord"]

# transform to bokeh data format
distSource = ColumnDataSource(distData)


#create figure for rank/value plot
distFig = figure(
            plot_width=300,
            plot_height=200,
            toolbar_location ="left",
            tools="lasso_select"
            )
# add glyphs to the figure 
distFig.circle(
    x="rank",
    y="value",
    fill_color = "blue",
    selection_fill_color = "red",
    selection_line_width = 2,
    selection_line_color = "red",
    fill_alpha = 0.05,
    line_width=0,
    muted_alpha = 0.05,
    size=4,
    name="distView",
    source=distSource
)


#create figure for map plot
mapFig = figure(
        plot_width=300,
        plot_height=200,
        toolbar_location ="left",
        tools="lasso_select"
    )

# add glyphs to map plot
mapper= LinearColorMapper(palette=Turbo256)
mapFig.circle(
    x="x_coord",
    y="y_coord",
    fill_color = {'field': 'value', 'transform': mapper},
    selection_fill_color = "red",
    fill_alpha = 1,
    line_width=0,
    muted_alpha = 0.4,
    size=4,
    name="mapView",
    source=distSource
)


# ==================================
# == callbacks: add interactivity ==

def updateDistCDS(selectionName):
    # re-create ColumnDataSource for dist & map figure
    distData = pd.DataFrame()
    distData["value"]= df[selectionName]
    distData["rank"] = get_rank(df[selectionName])
    distData["x_coord"] =  df["x_coord"]
    distData["y_coord"] = df["y_coord"]

    # update data
    distFig.select("distView").data_source.data = distData
    mapFig.select("mapView").data_source.data = distData
    
    # update titles
    distFig.title = selectionName + " distribution"
    mapFig.title = selectionName + " map"


# trigger a callback when a row on the table is selected 
def tableSelection_callback(attrname, old, new):
    # get row id
    selectionIndex=tableSource.selected.indices[0]
    # translate to column name
    selectionName= tableSource.data["index"][selectionIndex]
    # call functio to update plots
    updateDistCDS(selectionName)

tableSource.selected.on_change('indices', tableSelection_callback)



# ===================================================================
# Part II: run regression models, visualize and compare their results
# ===================================================================

# widgets for model parameter 
slider_TestSplit = Slider(start=0.01, 
                          end=0.81, 
                          value=0.1, 
                          step=0.02, 
                          title="Test Split", 
                          name="slider_TestSplit")

rb_TestSplitMode = RadioButtonGroup(labels=["selection", "random"], 
                                    active=0,
                                    name="rb_TestSplitMode")

compute_model= Button(label="compute",width =100, name="compute_model")


# create figure to 1) plot loss curves from training 
# 2) map view with dipicting error on test data
lossFig = figure(
            plot_width=600,
            plot_height=200,
            )

# dummy CDS
lossCDS = ColumnDataSource(data=(
    {"x_log":[0],
    "y_log":[0],
    "x_NN":[0],
    "y_NN":[0]
    }
))

lossFig.line(
    x="x_log",
    y="y_log",
    line_width= 1,
    line_color="orange",
    legend_label="log. reg.",
    name="logReg",
    source=lossCDS
)

lossFig.line(
    x="x_NN",
    y="y_NN",
    line_width= 1,
    line_color="blue",
    legend_label="NN.",
    name="NN",
    source=lossCDS
)



# ================================================================
# translate data-prep and regression code into a callback/function

def computeRegression(ts_slider, 
                        tsMode, 
                        dataCDS, 
                        df, 
                        OUTPUT_COLUMN='Accidents'):
    
    mbi_df = df.copy()
    
    # read-out user settings from UI widgets 
    tsMode = tsMode.active
    if tsMode == 0:
        # extract id's from selected datapoints from a ColumnDataSource
        testIDs = dataCDS.selected.indices
        test_df = mbi_df.iloc[testIDs]
    elif tsMode==1:
        # number of rows in a CDS
        test_df = mbi_df.sample(frac=ts_slider.value,random_state=0)

    train_df = mbi_df.drop(test_df.index)

    
    # Seperate input and output columns
    train_input_df = train_df.copy()
    train_output_df = train_input_df.pop(OUTPUT_COLUMN)

    test_input_df = test_df.copy()
    test_output_df = test_input_df.pop(OUTPUT_COLUMN)

    # Scalers
    input_scaler = MinMaxScaler()
    scaled_train_input_df = pd.DataFrame(
        input_scaler.fit_transform(train_input_df),
        columns=train_input_df.columns
    )
    scaled_test_input_df = input_scaler.transform(test_input_df)

    output_scaler = MinMaxScaler()
    scaled_train_output_df = pd.DataFrame(
        output_scaler.fit_transform(np.array(train_output_df).reshape(-1, 1)),
        columns=[OUTPUT_COLUMN]
    )
    scaled_test_output_df = output_scaler.transform(np.array(test_output_df).reshape(-1, 1))

    # Input shape
    input_shape = [*scaled_train_input_df.shape[1:]]

    # == Logistic regression ==================
    # Build model
    log_reg_model = keras.Sequential([
        layers.Dense(
            units=1,
            activation='sigmoid',
            input_shape=input_shape
        )
    ])

    # Compile model
    log_reg_model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
        loss=tf.keras.losses.MSE
    )

    # Train model
    log_reg_history = log_reg_model.fit(
        x=scaled_train_input_df,
        y=scaled_train_output_df,
        epochs=10,
        validation_split=0.2
    )

    # == neural network======================================
    # Build model
    neural_net_model = keras.Sequential([
        layers.Dense(
            units=128,
            activation='relu',
            input_shape=input_shape
        ),
        layers.Dense(units=64, activation='relu'),
        layers.Dense(units=32, activation='relu'),
        layers.Dense(units=1)
    ])
    neural_net_model.summary()

    # Compile model
    neural_net_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.MSE
    )

    # Train model
    neural_net_history = neural_net_model.fit(
        x=scaled_train_input_df,
        y=scaled_train_output_df,
        epochs=10,
        validation_split=0.2
    )


    # re-create CDS with new data and update lossCDS
    newData={}
    newData["y_log"] = log_reg_history.history['loss']
    newData["y_NN"] = neural_net_history.history['loss']
    newData["x_log"] = range(len(log_reg_history.history['loss']))
    newData["x_NN"] = range(len(neural_net_history.history['loss']))
    lossCDS.data = newData



# set-up callback
def cb_compute_model(event):
    computeRegression(slider_TestSplit,
                    rb_TestSplitMode, 
                    distSource,
                    df=df, 
                    OUTPUT_COLUMN = 'Accidents')

# assign callback to UI widget
compute_model.on_click(cb_compute_model)


# create layout
bokehLayout = row(data_table,
                    column(
                        row(distFig,mapFig),
                        row(slider_TestSplit,rb_TestSplitMode,compute_model),
                        lossFig),
                    name="bokehLayout")

# add to curDoc
curdoc().add_root(bokehLayout)

