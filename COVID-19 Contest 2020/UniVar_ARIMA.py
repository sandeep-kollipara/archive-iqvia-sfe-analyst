import numpy
import pandas
import matplotlib
import datetime
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import acf, pacf

# Data-prep 
data=pandas.read_csv(r"X:\Data\Interne_studies\Sandeep Kollipara\COVID-19 Contest 2020\COVID-19_Cases_World+India_20200324.csv");
data['Month']=data['Date'].apply(lambda x: x.split('/')[0]);
data['Day']=data['Date'].apply(lambda x: x.split('/')[1]);
data['Year']=data['Date'].apply(lambda x: x.split('/')[2]);
data['PyDate']=pandas.to_datetime({'year':data['Year'],
                                   'month':data['Month'],
                                   'day':data['Day']});
data.set_index('PyDate', inplace=True);
data_world=data[['World']];
data_india=data[['India']];

# Testing 'Stationarity'
matplotlib.pyplot.plot(data_world);
matplotlib.pyplot.plot(data_india);

def test_stationarity(timeseries):
    #Determing rolling statistics
    rolmean = timeseries.rolling(window=12).mean();
    rolstd = timeseries.rolling(window=12).std();
    #Plot rolling statistics:
    matplotlib.pyplot.plot(timeseries, color='blue',label='Original')
    matplotlib.pyplot.plot(rolmean, color='red', label='Rolling Mean')
    matplotlib.pyplot.plot(rolstd, color='black', label = 'Rolling Std')
    matplotlib.pyplot.legend(loc='best')
    matplotlib.pyplot.title('Rolling Mean & Standard Deviation')
    matplotlib.pyplot.show()
    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries.iloc[:,0].values, autolag='AIC');
    dfoutput = pandas.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)

test_stationarity(data_world);
test_stationarity(data_india);

# Logarithmic/Cube-root Transformation (log > cube-root) - Making the trend stationary
data_world_log = numpy.log(data_world);
matplotlib.pyplot.plot(data_world_log);
data_world_log=data_world_log.replace({numpy.inf:numpy.nan,
                                         -numpy.inf:numpy.nan}); # infinity values fix
data_world_log.dropna(inplace=True); # NaN fix
data_india_log = numpy.log(data_india);
matplotlib.pyplot.plot(data_india_log);
data_india_log=data_india_log.replace({numpy.inf:numpy.nan,
                                         -numpy.inf:numpy.nan}); # infinity values fix
data_india_log.dropna(inplace=True); # NaN fix

# Trend check by Smoothing + Aggression with Moving-average
world_moving_avg = data_world_log.rolling(window = 10).mean();
matplotlib.pyplot.plot(data_world_log);
matplotlib.pyplot.plot(world_moving_avg, color='red');
data_world_log_moving_avg_diff = data_world_log - world_moving_avg;
data_world_log_moving_avg_diff.dropna(inplace=True);
india_moving_avg = data_india_log.rolling(window = 10).mean();
matplotlib.pyplot.plot(data_india_log);
matplotlib.pyplot.plot(india_moving_avg, color='red');
data_india_log_moving_avg_diff = data_india_log - india_moving_avg;
data_india_log_moving_avg_diff.dropna(inplace=True);

# Re-test stationarity - Test statistic < Critical Value means SUCCESS
test_stationarity(data_world_log_moving_avg_diff);
test_stationarity(data_india_log_moving_avg_diff);

# Trend-check by Smoothing + Aggression with Exponentially-weighted-moving-average
world_exp_moving_avg = data_world_log.ewm(halflife=10).mean();
matplotlib.pyplot.plot(data_world_log);
matplotlib.pyplot.plot(world_exp_moving_avg, color='red');
data_world_log_exp_moving_avg_diff = data_world_log - world_exp_moving_avg;
data_world_log_exp_moving_avg_diff.dropna(inplace=True);
india_exp_moving_avg = data_india_log.ewm(halflife=10).mean();
matplotlib.pyplot.plot(data_india_log);
matplotlib.pyplot.plot(india_exp_moving_avg, color='red');
data_india_log_exp_moving_avg_diff = data_india_log - india_exp_moving_avg;
data_india_log_exp_moving_avg_diff.dropna(inplace=True);

# Re-test stationarity - this time with EWMA (Not as good as just MA)
test_stationarity(data_world_log_exp_moving_avg_diff);
test_stationarity(data_india_log_exp_moving_avg_diff);

# Seasonality-check by Differencing using time-lag
data_world_log_diff = data_world_log - data_world_log.shift();
matplotlib.pyplot.plot(data_world_log_diff);
data_world_log_diff.replace({numpy.inf:numpy.nan,
                             -numpy.inf:numpy.nan}, inplace=True); # infinity values fix
data_world_log_diff.dropna(inplace=True);
data_india_log_diff = data_india_log - data_india_log.shift();
matplotlib.pyplot.plot(data_india_log_diff);
data_india_log_diff.replace({numpy.inf:numpy.nan,
                             -numpy.inf:numpy.nan}, inplace=True); # infinity values fix
data_india_log_diff.dropna(inplace=True);

# Re-test stationarity - Good enough for India data but not World data
test_stationarity(data_world_log_diff);
test_stationarity(data_india_log_diff);

# Seasonality+Trend-check by Decomposing
world_decomposition = seasonal_decompose(data_world_log);
world_trend = world_decomposition.trend;
world_seasonal = world_decomposition.seasonal;
world_residual = world_decomposition.resid;
matplotlib.pyplot.subplot(411);
matplotlib.pyplot.plot(data_world_log, label='Original');
matplotlib.pyplot.legend(loc='best');
matplotlib.pyplot.subplot(412);
matplotlib.pyplot.plot(world_trend, label='Trend');
matplotlib.pyplot.legend(loc='best');
matplotlib.pyplot.subplot(413);
matplotlib.pyplot.plot(world_seasonal, label='Seasonality');
matplotlib.pyplot.legend(loc='best');
matplotlib.pyplot.subplot(414);
matplotlib.pyplot.plot(world_residual, label='Residuals');
matplotlib.pyplot.legend(loc='best');
matplotlib.pyplot.tight_layout();
data_world_log_decompose = pandas.DataFrame(world_residual);
data_world_log_decompose.dropna(inplace=True);
india_decomposition = seasonal_decompose(data_india_log);
india_trend = india_decomposition.trend;
india_seasonal = india_decomposition.seasonal;
india_residual = india_decomposition.resid;
matplotlib.pyplot.subplot(411);
matplotlib.pyplot.plot(data_india_log, label='Original');
matplotlib.pyplot.legend(loc='best');
matplotlib.pyplot.subplot(412);
matplotlib.pyplot.plot(india_trend, label='Trend');
matplotlib.pyplot.legend(loc='best');
matplotlib.pyplot.subplot(413);
matplotlib.pyplot.plot(india_seasonal, label='Seasonality');
matplotlib.pyplot.legend(loc='best');
matplotlib.pyplot.subplot(414);
matplotlib.pyplot.plot(india_residual, label='Residuals');
matplotlib.pyplot.legend(loc='best');
matplotlib.pyplot.tight_layout();
data_india_log_decompose = pandas.DataFrame(india_residual);
data_india_log_decompose.dropna(inplace=True);

# Re-test stationarity - World & India data 99%+ confident stationary
test_stationarity(data_world_log_decompose);
test_stationarity(data_india_log_decompose);

# Pre-model parameter setup - World P=3(PACF), Q=8(ACF); India P=1, Q=1;
world_lag_acf = acf(data_world_log_diff, nlags=10, fft=False);
world_lag_pacf = pacf(data_world_log_diff, nlags=10, method='ols');
matplotlib.pyplot.subplot(121);
matplotlib.pyplot.plot(world_lag_acf)
matplotlib.pyplot.axhline(y=0,linestyle='--',color='gray');
matplotlib.pyplot.axhline(y=-1.96/numpy.sqrt(len(data_world_log_diff)),linestyle='--',color='gray');
matplotlib.pyplot.axhline(y=1.96/numpy.sqrt(len(data_world_log_diff)),linestyle='--',color='gray');
matplotlib.pyplot.title('Auto-Correlation Function');
matplotlib.pyplot.subplot(122);
matplotlib.pyplot.plot(world_lag_pacf)
matplotlib.pyplot.axhline(y=0,linestyle='--',color='gray');
matplotlib.pyplot.axhline(y=-1.96/numpy.sqrt(len(data_world_log_diff)),linestyle='--',color='gray');
matplotlib.pyplot.axhline(y=1.96/numpy.sqrt(len(data_world_log_diff)),linestyle='--',color='gray');
matplotlib.pyplot.title('Partial Auto-Correlation Function');
matplotlib.pyplot.tight_layout();
india_lag_acf = acf(data_india_log_diff, nlags=10, fft=False);
india_lag_pacf = pacf(data_india_log_diff, nlags=10, method='ols');
matplotlib.pyplot.subplot(121);
matplotlib.pyplot.plot(india_lag_acf)
matplotlib.pyplot.axhline(y=0,linestyle='--',color='gray');
matplotlib.pyplot.axhline(y=-1.96/numpy.sqrt(len(data_india_log_diff)),linestyle='--',color='gray');
matplotlib.pyplot.axhline(y=1.96/numpy.sqrt(len(data_india_log_diff)),linestyle='--',color='gray');
matplotlib.pyplot.title('Auto-Correlation Function');
matplotlib.pyplot.subplot(122);
matplotlib.pyplot.plot(india_lag_pacf)
matplotlib.pyplot.axhline(y=0,linestyle='--',color='gray');
matplotlib.pyplot.axhline(y=-1.96/numpy.sqrt(len(data_india_log_diff)),linestyle='--',color='gray');
matplotlib.pyplot.axhline(y=1.96/numpy.sqrt(len(data_india_log_diff)),linestyle='--',color='gray');
matplotlib.pyplot.title('Partial Auto-Correlation Function');
matplotlib.pyplot.tight_layout();

# AR Model
world_model = ARIMA(data_world_log, order=(3,1,0)); # order= (P, D, Q)
world_results_AR = world_model.fit(disp=1);
matplotlib.pyplot.plot(data_world_log_diff);
matplotlib.pyplot.plot(world_results_AR.fittedvalues, color='red');
matplotlib.pyplot.title('RSS: %.4f'% numpy.nansum((world_results_AR.fittedvalues - data_world_log_diff.iloc[:,0])**2));
india_model = ARIMA(data_india_log, order=(1,1,0)); # order= (P, D, Q)
india_results_AR = india_model.fit(disp=1);
matplotlib.pyplot.plot(data_india_log_diff);
matplotlib.pyplot.plot(india_results_AR.fittedvalues, color='red');
matplotlib.pyplot.title('RSS: %.4f'% numpy.nansum((india_results_AR.fittedvalues - data_india_log_diff.iloc[:,0])**2));

# MA Model
world_model = ARIMA(data_world_log, order=(0,1,8)); # order= (P, D, Q)
world_results_MA = world_model.fit(disp=1);
matplotlib.pyplot.plot(data_world_log_diff);
matplotlib.pyplot.plot(world_results_MA.fittedvalues, color='red');
matplotlib.pyplot.title('RSS: %.4f'% numpy.nansum((world_results_MA.fittedvalues - data_world_log_diff.iloc[:,0])**2));
india_model = ARIMA(data_india_log, order=(0,1,1)); # order= (P, D, Q)
india_results_MA = india_model.fit(disp=1);
matplotlib.pyplot.plot(data_india_log_diff);
matplotlib.pyplot.plot(india_results_MA.fittedvalues, color='red');
matplotlib.pyplot.title('RSS: %.4f'% numpy.nansum((india_results_MA.fittedvalues - data_india_log_diff.iloc[:,0])**2));

# ARIMA Model - World model: Q value reduced from 8 to 5 since stationarity is lost at higher Q values
world_model = ARIMA(data_world_log, order=(3,1,5)); # order= (P, D, Q)
world_results_ARIMA = world_model.fit(disp=1);
matplotlib.pyplot.plot(data_world_log_diff);
matplotlib.pyplot.plot(world_results_ARIMA.fittedvalues, color='red');
matplotlib.pyplot.title('RSS: %.4f'% numpy.nansum((world_results_ARIMA.fittedvalues - data_world_log_diff.iloc[:,0])**2));
india_model = ARIMA(data_india_log, order=(1,1,1)); # order= (P, D, Q)
india_results_ARIMA = india_model.fit(disp=1);
matplotlib.pyplot.plot(data_india_log_diff);
matplotlib.pyplot.plot(india_results_ARIMA.fittedvalues, color='red');
matplotlib.pyplot.title('RSS: %.4f'% numpy.nansum((india_results_ARIMA.fittedvalues - data_india_log_diff.iloc[:,0])**2));

#Forecasting
#world_results_ARIMA.forecast()[1];
world_forecast=world_results_ARIMA.predict(start=datetime.date(2020,3,24), end=datetime.date(2020,4,3));
india_forecast=india_results_ARIMA.predict(start=datetime.date(2020,3,24), end=datetime.date(2020,4,3));

# Reverting results
world_predictions_ARIMA_diff_original = pandas.Series(world_results_ARIMA.fittedvalues, copy=True);
world_predictions_ARIMA_diff = world_predictions_ARIMA_diff_original.append(world_forecast, verify_integrity=True, ignore_index=False);
world_predictions_ARIMA_diff_cumsum = world_predictions_ARIMA_diff.cumsum();
#world_predictions_ARIMA_log = pandas.Series(data_world_log['World'].iloc[0], index=data_world_log.index);
world_predictions_ARIMA_log = pandas.Series(data_world_log['World'].iloc[0], index=world_predictions_ARIMA_diff_cumsum.index);
world_predictions_ARIMA_log = world_predictions_ARIMA_log.add(world_predictions_ARIMA_diff_cumsum, fill_value=0);
world_predictions_ARIMA = numpy.exp(world_predictions_ARIMA_log);
matplotlib.pyplot.plot(data_world);
matplotlib.pyplot.plot(world_predictions_ARIMA);
matplotlib.pyplot.title('RMSE: %.4f'% numpy.sqrt(sum((world_predictions_ARIMA[:len(data_world_log)-1] - data_world.loc[data_world_log.index[1:], 'World'])**2)/(len(data_world)-1)));
india_predictions_ARIMA_diff_original = pandas.Series(india_results_AR.fittedvalues, copy=True);
india_predictions_ARIMA_diff = india_predictions_ARIMA_diff_original.append(india_forecast, verify_integrity=True, ignore_index=False);
india_predictions_ARIMA_diff_cumsum = india_predictions_ARIMA_diff.cumsum();
#world_predictions_ARIMA_log = pandas.Series(data_world_log['World'].iloc[0], index=data_world_log.index);
india_predictions_ARIMA_log = pandas.Series(data_india_log['India'].iloc[0], index=india_predictions_ARIMA_diff_cumsum.index);
india_predictions_ARIMA_log = india_predictions_ARIMA_log.add(india_predictions_ARIMA_diff_cumsum, fill_value=0);
india_predictions_ARIMA = numpy.exp(india_predictions_ARIMA_log);
matplotlib.pyplot.plot(data_india);
matplotlib.pyplot.plot(india_predictions_ARIMA);
matplotlib.pyplot.title('RMSE: %.4f'% numpy.sqrt(sum((india_predictions_ARIMA[:len(data_india_log)-1] - data_india.loc[data_india_log.index[1:], 'India'])**2)/(len(data_india)-1)));

#Results
print(world_predictions_ARIMA[-11:]);
print(india_predictions_ARIMA[-11:]);
