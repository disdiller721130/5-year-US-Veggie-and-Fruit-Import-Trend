# Dependencies
import os

import pandas as pd
import numpy as np

import sqlalchemy
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, func
import datetime as dt
import numpy as np
from flask import Flask, render_template, jsonify
from jinja2 import Template
from flask_sqlalchemy import SQLAlchemy

import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Activation
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import backend

app = Flask(__name__)


#################################################
# Database Setup
#################################################

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///db/Veggie_Fruit_DB.sqlite"
db = SQLAlchemy(app)

Base = automap_base()
# reflect the tables
Base.prepare(db.engine, reflect=True)

FED = Base.classes.fruitexportdest
FEV = Base.classes.fruitexportval
FIS = Base.classes.fruitimportsource
FIV = Base.classes.fruitimportval
VED = Base.classes.veggieexportdest
VEV = Base.classes.veggieexportval
VIS = Base.classes.veggieimportsource
VIV = Base.classes.veggieimportval

@app.route("/")
def index():
    """Return the homepage."""
    return render_template("full_index.html")

@app.route("/map")
def map():
    return render_template("map_index.html")

@app.route("/summary_edata")
def summary_edata():

    fe_summary = db.session.query(FED.country,FED.yr17,FED.yr18).group_by(FED.country).order_by(FED.yr18.desc()).all()
    fe_summary_df = pd.DataFrame(fe_summary, columns=['country', 'yr17', 'yr18'])
    fe_summary_df['yr17'] = fe_summary_df['yr17'].str.replace(',', '')
    fe_summary_df['yr18'] = fe_summary_df['yr18'].str.replace(',', '')
    fe_summary_df=fe_summary_df.dropna(how="any")
    fe_summary_df["yr17"] = pd.to_numeric(fe_summary_df["yr17"])
    fe_summary_df["yr18"] = pd.to_numeric(fe_summary_df["yr18"])

    ve_summary = db.session.query(VED.country,VED.yr17,VED.yr18).group_by(VED.country).order_by(VED.yr18.desc()).all()
    ve_summary_df = pd.DataFrame(ve_summary, columns=['country', 'yr17', 'yr18'])
    ve_summary_df['yr17'] = ve_summary_df['yr17'].str.replace(',', '')
    ve_summary_df['yr18'] = ve_summary_df['yr18'].str.replace(',', '')
    ve_summary_df=ve_summary_df.dropna(how="any")
    ve_summary_df["yr17"] = pd.to_numeric(ve_summary_df["yr17"])
    ve_summary_df["yr18"] = pd.to_numeric(ve_summary_df["yr18"])
    esummary_frames = [ve_summary_df, fe_summary_df]
    summary_export = pd.concat(esummary_frames,ignore_index=False)
    summary_export = summary_export.sort_values(['yr17', 'yr18'], ascending=False)
    export_yr17sum=summary_export['yr17'].sum()
    export_yr18sum=summary_export['yr18'].sum()

    for i in summary_export:
        e_ms17=summary_export['yr17']/export_yr17sum*100
        e_ms18=summary_export['yr18']/export_yr18sum*100
 
    summary_export["MarketShare(2017)"]=e_ms17
    summary_export["MarketShare(2018)"]=e_ms18
    summary_export["ChangeInMarketShare"]=e_ms18-e_ms17
    renamed_summary_export = summary_export.rename(columns={"country":"Country", "MS 2017":"MarketShare(2017)","MS 2018":"MarketShare(2018)", "Change in MS":"ChangeInMarketShare","yr17":"Value(2017)", "yr18":"Value(2018)"})
    summary_export = renamed_summary_export[["Country","MarketShare(2017)","MarketShare(2018)","ChangeInMarketShare", "Value(2017)", "Value(2018)"]]
    summary_export=summary_export.sort_values(['MarketShare(2018)'], ascending=False).head()
    Summary_Export = summary_export[["Country", "MarketShare(2017)", "MarketShare(2018)", "ChangeInMarketShare", "Value(2017)", "Value(2018)"]]
    summary_export_list = []

    Summary_Export = Summary_Export.reset_index(drop=True)
    ms17_list = Summary_Export["MarketShare(2017)"].tolist()
    ms18_list = Summary_Export["MarketShare(2018)"].tolist()
    mschange_list = Summary_Export["ChangeInMarketShare"].tolist()
    value_yr17 = Summary_Export["Value(2017)"].tolist()
    value_yr18 = Summary_Export["Value(2018)"].tolist()

    country_list = Summary_Export["Country"].tolist()
    for i in range(len(country_list)):
        summary_report_dict = {}
        summary_report_dict["Country"] = country_list[i]
        summary_report_dict["MS17"] = round(ms17_list[i],1)
        summary_report_dict["MS18"] = round(ms18_list[i],1)
        summary_report_dict["MSChange"] = round(mschange_list[i],1)
        summary_report_dict["Value_yr17"] = value_yr17[i]
        summary_report_dict["Value_yr18"] = value_yr18[i]
    
        summary_export_list.append(summary_report_dict)

    print(summary_export_list)
    return jsonify(summary_export_list)


@app.route("/summary_idata")
def summary_idata():

    fi_summary = db.session.query(FIS.country,FIS.yr17,FED.yr18).group_by(FIS.country).all()
    fi_summary_df = pd.DataFrame(fi_summary, columns=['country', 'yr17', 'yr18'])
    fi_summary_df['yr17'] = fi_summary_df['yr17'].str.replace(',', '')
    fi_summary_df['yr18'] = fi_summary_df['yr18'].str.replace(',', '')
    fi_summary_df=fi_summary_df.dropna(how="any")
    fi_summary_df["yr17"] = pd.to_numeric(fi_summary_df["yr17"])
    fi_summary_df["yr18"] = pd.to_numeric(fi_summary_df["yr18"])

    vi_summary = db.session.query(VIS.country,VIS.yr17,VIS.yr18).group_by(VIS.country).all()
    vi_summary_df = pd.DataFrame(vi_summary, columns=['country', 'yr17', 'yr18'])
    vi_summary_df['yr17'] = vi_summary_df['yr17'].str.replace(',', '')
    vi_summary_df['yr18'] = vi_summary_df['yr18'].str.replace(',', '')
    vi_summary_df=vi_summary_df.dropna(how="any")
    vi_summary_df["yr17"] = pd.to_numeric(vi_summary_df["yr17"])
    vi_summary_df["yr18"] = pd.to_numeric(vi_summary_df["yr18"])

    isummary_frames = [vi_summary_df, fi_summary_df]
    summary_import = pd.concat(isummary_frames,ignore_index=False)
    summary_import["Total"] = summary_import["yr17"] + summary_import["yr18"]
    summary_import = summary_import.sort_values(["Total"], ascending=False)
    import_yr17sum=summary_import['yr17'].sum()
    import_yr18sum=summary_import['yr18'].sum()

    i_ms17=summary_import['yr17']/import_yr17sum*100
    i_ms18=summary_import['yr18']/import_yr18sum*100
    summary_import["MarketShare(2017)"]=i_ms17
    summary_import["MarketShare(2018)"]=i_ms18
    summary_import["ChangeInMarketShare"] = i_ms18-i_ms17
    renamed_summary_import = summary_import.rename(columns={"country":"Country", "MS 2017":"MarketShare(2017)",
                                                        "MS 2018":"MarketShare(2018)", "Change in MS":"ChangeInMarketShare",
                                                        "yr17":"Value(2017)", "yr18":"Value(2018)"})
    summary_import_list = []

    renamed_summary_import = renamed_summary_import.reset_index(drop=True)
    ms17_list = renamed_summary_import["MarketShare(2017)"].tolist()
    ms18_list = renamed_summary_import["MarketShare(2018)"].tolist()
    mschange_list = renamed_summary_import["ChangeInMarketShare"].tolist()
    value_yr17 = renamed_summary_import["Value(2017)"].tolist()
    value_yr18 = renamed_summary_import["Value(2018)"].tolist()
    country_list = renamed_summary_import["Country"].tolist()

    for i in range(len(country_list)):
        summary_report_dict = {}
        summary_report_dict["Country"] = country_list[i]
        summary_report_dict["MS17"] = round(ms17_list[i],1)
        summary_report_dict["MS18"] = round(ms18_list[i],1)
        summary_report_dict["MSChange"] = round(mschange_list[i],1)
        summary_report_dict["Value_yr17"] = value_yr17[i]
        summary_report_dict["Value_yr18"] = value_yr18[i]
    
        summary_import_list.append(summary_report_dict)

    return jsonify(summary_import_list)

@app.route("/top10veg_edata")
def top10veg_edata(): 

    ExportDataVeggie = db.session.query(VED.country,VED.product,VED.yr17,VED.yr18).\
    order_by(VED.country.asc()).all()

    ve_data_df = pd.DataFrame(ExportDataVeggie, columns=['country','product', 'yr17', 'yr18'])

    ve_data_df['yr17'] = ve_data_df['yr17'].str.replace(',', '')
    ve_data_df['yr18'] = ve_data_df['yr18'].str.replace(',', '')
    ve_data_df = ve_data_df.dropna(how="any")
    ve_data_df["yr17"] = pd.to_numeric(ve_data_df["yr17"])
    ve_data_df["yr18"] = pd.to_numeric(ve_data_df["yr18"])
    ve_data_df["average"] = (ve_data_df["yr17"] + ve_data_df["yr18"])/2
    ve_data_df = ve_data_df.sort_values(["average"], ascending=False)
    ve_data_result = ve_data_df.reset_index(drop=True)
    ve_data_result = ve_data_result[:10]

    country_list = ve_data_result["country"].tolist()
    product_list = ve_data_result["product"].tolist()
    avg_list = ve_data_result["average"].tolist()
    veggie_yr17 = ve_data_result["yr17"].tolist()
    veggie_yr18 = ve_data_result["yr18"].tolist()
    vegexportdest_list = []

    for i in range(len(country_list)):
        vegexportdest_dict = {}
        vegexportdest_dict["Country"] = country_list[i]
        vegexportdest_dict["Product"] = product_list[i]
        vegexportdest_dict["YR17"] = veggie_yr17[i]
        vegexportdest_dict["YR18"] = veggie_yr18[i]
        vegexportdest_dict["Average"] = avg_list[i]
    
        vegexportdest_list.append(vegexportdest_dict)
       
    return jsonify(vegexportdest_list)

@app.route("/top10fruit_edata")
def top10fruit_edata(): 

    sel = [
    FED.country,
    FED.product,
    FED.yr17,
    FED.yr18,
    ]
    ExportDataFruit = db.session.query(*sel).order_by(FED.country.desc()).all()
    fe_data_df = pd.DataFrame(ExportDataFruit, columns=['country','product', 'yr17', 'yr18'])
    fe_data_df["yr17"] = fe_data_df["yr17"].str.replace(",","")
    fe_data_df["yr18"] = fe_data_df["yr18"].str.replace(",","")
    fe_data_df = fe_data_df.dropna(how = 'any')
    fe_data_df["yr17"] = pd.to_numeric(fe_data_df["yr17"])
    fe_data_df["yr18"] = pd.to_numeric(fe_data_df["yr18"])
    fe_data_df["average"] = (fe_data_df["yr17"] + fe_data_df["yr18"])/2
    fe_data_df = fe_data_df.sort_values(["average"], ascending=False)
    fe_data_result = fe_data_df.reset_index(drop=True)
    fe_data_result = fe_data_result[:10]

    country_list = fe_data_result["country"].tolist()
    product_list = fe_data_result["product"].tolist()
    avg_list = fe_data_result["average"].tolist()
    fruit_yr17 = fe_data_result["yr17"].tolist()
    fruit_yr18 = fe_data_result["yr18"].tolist()
    fruitexportdest_list = []

    for i in range(len(country_list)):
        fruitexportdest_dict = {}
        fruitexportdest_dict["Country"] = country_list[i]
        fruitexportdest_dict["Product"] = product_list[i]
        fruitexportdest_dict["YR17"] = fruit_yr17[i]
        fruitexportdest_dict["YR18"] = fruit_yr17[i]
        fruitexportdest_dict["Average"] = avg_list[i]
    
        fruitexportdest_list.append(fruitexportdest_dict)
       
    return jsonify(fruitexportdest_list)
 
@app.route("/top10veg_edata_bar")
def top10veg_data_bar():

    ExportDataVeggie = db.session.query(VED.country,VED.product,VED.yr17,VED.yr18).\
    order_by(VED.country.asc()).all()

    ve_data_df = pd.DataFrame(ExportDataVeggie, columns=['country','product', 'yr17', 'yr18'])

    ve_data_df['yr17'] = ve_data_df['yr17'].str.replace(',', '')
    ve_data_df['yr18'] = ve_data_df['yr18'].str.replace(',', '')
    ve_data_df = ve_data_df.dropna(how="any")
    ve_data_df["yr17"] = pd.to_numeric(ve_data_df["yr17"])
    ve_data_df["yr18"] = pd.to_numeric(ve_data_df["yr18"])
    ve_data_df["average"] = (ve_data_df["yr17"] + ve_data_df["yr18"])/2
    ve_data_df = ve_data_df.sort_values(["average"], ascending=False)
    ve_data_result = ve_data_df.reset_index(drop=True)
    
    ve_summary = ve_data_result.groupby(['product'])
    ve_summary_df = pd.DataFrame({'Veggie_Export': ve_summary['average'].sum()})
    ve_summary_df = ve_summary_df.sort_values(['Veggie_Export'], ascending = False)
    product_list = ve_summary_df.index.tolist()
    ve_top10 = product_list[:10]
    ve_top10_qty = ve_summary_df['Veggie_Export'][:10].tolist()
    data = {'OverallExport': ve_top10, 'Average': ve_top10_qty}

    return jsonify(data)

@app.route("/top10fruit_edata_bar")
def top10fruit_data_bar():

    ExportDataFruit = db.session.query(FED.country,FED.product,FED.yr17,FED.yr18).\
    order_by(FED.country.asc()).all()

    fe_data_df = pd.DataFrame(ExportDataFruit, columns=['country','product', 'yr17', 'yr18'])

    fe_data_df['yr17'] = fe_data_df['yr17'].str.replace(',', '')
    fe_data_df['yr18'] = fe_data_df['yr18'].str.replace(',', '')
    fe_data_df = fe_data_df.dropna(how="any")
    fe_data_df["yr17"] = pd.to_numeric(fe_data_df["yr17"])
    fe_data_df["yr18"] = pd.to_numeric(fe_data_df["yr18"])
    fe_data_df["average"] = (fe_data_df["yr17"] + fe_data_df["yr18"])/2
    fe_data_df = fe_data_df.sort_values(["average"], ascending=False)
    fe_data_result = fe_data_df.reset_index(drop=True)
    
    fe_summary = fe_data_result.groupby(['product'])
    fe_summary_df = pd.DataFrame({'Fruit_Export': fe_summary['average'].sum()})
    fe_summary_df = fe_summary_df.sort_values(['Fruit_Export'], ascending = False)
    product_list = fe_summary_df.index.tolist()
    fe_top10 = product_list[:10]
    fe_top10_qty = fe_summary_df['Fruit_Export'][:10].tolist()
    data = {'OverallExport': fe_top10, 'Average': fe_top10_qty}

    return jsonify(data)

@app.route("/topfruit_idata")
def topfruit_idata():  

    ImportDataFruit = db.session.query(FIS.country,FIS.product,FIS.yr17,VED.yr18).\
    order_by(FIS.country.asc()).all()

    fi_data_df = pd.DataFrame(ImportDataFruit, columns=['country','product', 'yr17', 'yr18'])
    fi_data_df['yr17'] = fi_data_df['yr17'].str.replace(',', '')
    fi_data_df['yr18'] = fi_data_df['yr18'].str.replace(',', '')
    fi_data_df = fi_data_df.dropna(how="any")
    fi_data_df["yr17"] = pd.to_numeric(fi_data_df["yr17"])
    fi_data_df["yr18"] = pd.to_numeric(fi_data_df["yr18"])
    fi_data_df["average"] = (fi_data_df["yr17"] + fi_data_df["yr18"])/2
    fi_data_df = fi_data_df.sort_values(["average"], ascending=False)
    fi_data_result = fi_data_df.reset_index(drop=True)
    fi_data_result = fi_data_result[:10]

    country_list = fi_data_result["country"].tolist()
    product_list = fi_data_result["product"].tolist()
    avg_list = fi_data_result["average"].tolist() 
    fruit_yr17 = fi_data_result["yr17"].tolist()
    fruit_yr18 = fi_data_result["yr18"].tolist()
    fruitimportsource_list = []

    for i in range(len(country_list)):
        fruitimportsource_dict = {}
        fruitimportsource_dict["Country"] = country_list[i]
        fruitimportsource_dict["Product"] = product_list[i]
        fruitimportsource_dict["YR17"] = fruit_yr17[i]
        fruitimportsource_dict["YR18"] = fruit_yr18[i]
        fruitimportsource_dict["Average"] = avg_list[i]
    
        fruitimportsource_list.append(fruitimportsource_dict)

    return jsonify(fruitimportsource_list)

@app.route("/veggienames")
def vegnames():
    
    market_year = db.session.query(VED.product , VED.country , VED.share).all()
    mkt_yr_df = pd.DataFrame(market_year, columns = ["Product","Country","Share"])
    product_group = mkt_yr_df.groupby(["Product"]).count()
    product_list = product_group.index.values.tolist()
    product_list = product_list[1:]

    return jsonify(product_list)

@app.route("/fruitnames")
def fruitnames():
    """Return a list of sample names."""
    
    fruit_specific = db.session.query(FED.product , FED.country , FED.share).all()
    fruit_spec_df = pd.DataFrame(fruit_specific, columns = ["Product","Country","Share"])
    fruit_group = fruit_spec_df.groupby(["Product"]).count()
    fruit_overall = fruit_group.index.values.tolist()

    return jsonify(fruit_overall)

@app.route("/veggieexp/<veggie>")
def veg_country(veggie):

    sel = [
        VED.product,
        VED.country,
        VED.share,
        VED.lat,
        VED.lon,
    ]
    results = db.session.query(*sel).filter(VED.product == veggie).all()
    veggie_list = []
    for result in results:
        veggie_country = {}
        veggie_country["Country"] = result[1]
        veggie_country["Type"] = "veggie"
        veggie_country["Share"] = float(result[2].replace('%',''))
        veggie_country["lat"] = result[3]
        veggie_country["lon"] = result[4]
        veggie_list.append(veggie_country)

    return jsonify(veggie_list)

@app.route("/fruitexp/<fruit>")
def fruit_country(fruit):

    sel = [
        FED.product,
        FED.country,
        FED.share,
        FED.lat,
        FED.lon,
    ]
    results = db.session.query(*sel).filter(FED.product == fruit).all()
    fruit_list = []
    for result in results:
        fruit_country = {}
        fruit_country["Country"] = result[1]
        fruit_country["Type"] = "Fruit"
        fruit_country["Share"] = float(result[2].replace('%',''))
        fruit_country["lat"] = result[3]
        fruit_country["lon"] = result[4]
        fruit_list.append(fruit_country)

    return jsonify(fruit_list)

@app.route("/product/<product>")
def product_prediction(product):

    fev_summary = db.session.query(FEV).statement
    fev_summary_df = pd.read_sql_query(fev_summary, db.session.bind)
    fev_df = fev_summary_df.drop('id', axis = 1)
    fev_df = fev_df.replace({',':''}, regex = True)
    fev_df = fev_df.dropna(how = 'any')
    fev_number = fev_df.columns[3:]
    for i in range(0, len(fev_number)):
        fev_df[fev_number[i]] = pd.to_numeric(fev_df[fev_number[i]])
    
    vev_summary = db.session.query(VEV).order_by(VEV.product.asc()).statement
    vev_df = pd.read_sql_query(vev_summary, db.session.bind)
    vev_df = vev_df.drop('id', axis = 1)
    vev_df = vev_df.replace({',':''}, regex = True)
    vev_df = vev_df.dropna(how = 'any')
    vev_number = vev_df.columns[3:]
    for i in range(0, len(vev_number)):
        vev_df[vev_number[i]] = pd.to_numeric(vev_df[vev_number[i]])
    
    e_frames = [vev_df, fev_df]
    export_all = pd.concat(e_frames,ignore_index=False)

    export_specific = export_all.set_index("product")
    product_overall = export_specific.loc[product, :]
    product_specific = product_overall.reset_index(drop = True)

    product_month_list = []

    result_2019 = product_specific.iloc[0,3:6]

    length = len(product_specific.mktyr) - 1
    while length > 0:
        result_rest = product_specific.iloc[length,3:15]
        for r in range(0, len(result_rest)):
            product_month_list.append(result_rest[r])
        length -= 1

    #2019 first 3 months:
    for i in range(0, len(result_2019)):
        product_month_list.append(result_2019[i])

    month = [m for m in range(1, len(product_month_list) + 1)]
    product_export_df = pd.DataFrame({"month": month, product: product_month_list})

    training_set = product_export_df[product].values.reshape(-1, 1)

    X_scaler = MinMaxScaler().fit(training_set)
    training_set_scaled = X_scaler.transform(training_set)

    X_train = []
    y_train = []

    for i in range(5, 51):
        X_train.append(training_set_scaled[i-5:i])
        y_train.append(training_set_scaled[i])
    X_train, y_train = np.array(X_train), np.array(y_train)

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    X_test = []
    y_test = []

    for i in range(5, 21):
        X_test.append(training_set_scaled[i-5:i])
        y_test.append(training_set_scaled[i])
    X_test, y_test = np.array(X_test), np.array(y_test)

    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    model = Sequential()
    model.add(LSTM(10, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(24, return_sequences=False))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer='rmsprop')
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs = 20, batch_size = 10)
    
    predicted_product_price = model.predict(X_test)
    product_prediction = X_scaler.inverse_transform(predicted_product_price)
    product_price = X_scaler.inverse_transform(y_test)
    model_loss = model.evaluate(X_test, y_test, verbose=2)
    
    product_export_summary = {}
    product_export_summary["Prediction Price"] = round(float(product_prediction[5]),2)
    product_export_summary["Actual Price"] = round(float(product_price[5]),2)
    product_export_summary["MSE"] = round(model_loss,3)
    print(product_export_summary)
    
    return jsonify(product_export_summary)

if __name__ == "__main__":
    app.run()

