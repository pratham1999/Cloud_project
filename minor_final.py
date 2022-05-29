import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import io
import urllib, base64
from IPython import get_ipython
from flask import Flask,request, render_template

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0


@app.route('/')
def project():
    return render_template("main.html")

@app.route('/predict',methods=['POST','GET'])
def predict():
  if request.method == "POST": 

      df = pd.read_csv("NIFTY50_all.csv",parse_dates=['Date'])

      company=request.form.get('companies')
      df1=df.loc[df['Symbol']==company]
      df1=df1[['Date','Open','High','Low','Close','Volume']]


      df1.reset_index()
      df1.set_index('Date',inplace=True)

      #converting to base64
      def base64_convert(fig):
            img = io.BytesIO()
            fig.savefig(img, format='png',
                bbox_inches='tight')
            img.seek(0)

            return base64.b64encode(img.getvalue())

    #plotting closing price for the company

      plt.plot(df1['Close'])
      plt.xlabel('Date')
      plt.ylabel('Closing Price')
      plt.title(company)
      fig=plt.gcf()
      plt.show()

      data_encoded=base64_convert(fig)
      data_html = '<img src="data:image/png;base64, {}" />'.format(data_encoded.decode('utf-8'))


      #normalization
      from sklearn.preprocessing import MinMaxScaler
      scaler=MinMaxScaler(feature_range=(0,1))
      df1=scaler.fit_transform(df1)

      #Spliting train and test data
      training_size=int(len(df1)*0.75)
      test_size=len(df1)-training_size
      train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:]

      def create_dataset(dataset,time_step=1):
          dataX=[]
          dataY=[]
          for i in range(len(dataset)-time_step-1):
              a = dataset[i:(i+time_step)]
              dataX.append(a)
              dataY.append(dataset[i+time_step,:])
          return np.array(dataX), np.array(dataY)

      time_step = 100
      X_train, Y_train = create_dataset(train_data,time_step)
      X_test, Y_test = create_dataset(test_data, time_step)

      x=company +".h5"
      from keras.models import load_model
      model = load_model(x)

      train_predict=model.predict(X_train)
      test_predict=model.predict(X_test)

      train_predict = scaler.inverse_transform(train_predict)
      test_predict = scaler.inverse_transform(test_predict)

      import math
      from sklearn.metrics import mean_squared_error
      math.sqrt(mean_squared_error(Y_train[:,3],train_predict[:,3]))
      math.sqrt(mean_squared_error(Y_test[:,3],test_predict[:,3]))

      df2=scaler.inverse_transform(df1)
      df2 = df2[:,3]
      df2 = np.reshape(df2, (-1, 1))
      train_predict=train_predict[:,3]
      test_predict=test_predict[:,3]
      train_predict=np.reshape(train_predict,(-1,1))
      test_predict=np.reshape(test_predict,(-1,1))
      plt.plot(df2)
      look_back=100
      trainPredictPlot=np.empty_like(df2)
      trainPredictPlot[:,:]=np.nan
      trainPredictPlot[look_back:len(train_predict)+look_back,:] = train_predict
      testPredictPlot=np.empty_like(df2)
      testPredictPlot[:,:]=np.nan
      testPredictPlot[len(train_predict)+(look_back*2)+1:len(df2)-1,:] = test_predict
      plt.plot(trainPredictPlot)
      plt.plot(testPredictPlot)
      plt.xlabel('Days')
      plt.ylabel('Closing Price')
      labels=['actual','train','test']
      plt.legend(labels)
      plt.title(company)
      fig=plt.gcf()
      plt.show()
      train_encoded=base64_convert(fig)
      train_html = '<img src="data:image/png;base64, {}" />'.format(train_encoded.decode('utf-8'))

      x_input=test_data[-100:].reshape(1,-1,5)
      temp_input=list(x_input)
      temp_input=temp_input[0].tolist()

      lst_output=[]
      n_steps=100
      i=0
      while(i<100):
            
            if(len(temp_input)>100):
                #print(temp_input)
                x_input=np.array(temp_input[1:])
                print("{} day".format(i+1))
                #print(x_input)
                x_input = x_input.reshape((1, n_steps, 5))
        
                yhat = model.predict(x_input, verbose=0)
                print("data: {}".format(yhat))
                temp_input.extend(yhat.tolist())
                temp_input=temp_input[1:]
                #print(temp_input)
                lst_output.extend(yhat.tolist())
                i=i+1
            else:
                x_input = x_input.reshape((1, n_steps,5))
                yhat= model.predict(x_input, verbose=0)
                print("{} day".format(i+1))
                print(yhat[:,3])
                temp_input.extend(yhat.tolist())
                print(len(temp_input))
                lst_output.extend(yhat.tolist())
                i=i+1
    

      #print(lst_output)

      lst_output=scaler.inverse_transform(lst_output)
      pred_closing=lst_output[:,3]

      day_new=np.arange(1,101)
      day_pred=np.arange(101,201)

      plt.plot(day_new,df2[-100:])
      plt.plot(day_pred,pred_closing)
      plt.xlabel('Days')
      plt.ylabel('Closing Price')
      labels=['actual','predicted']
      plt.legend(labels)
      plt.title(company)
      fig=plt.gcf()
      plt.show()
      prediction_encoded=base64_convert(fig)
      prediction_html = '<img src="data:image/png;base64, {}" />'.format(prediction_encoded.decode('utf-8'))

      return render_template('last.html',data_graph=data_encoded.decode('utf-8'),desc1='Closing Price of Historical Data',train_graph=train_encoded.decode('utf-8'),desc2='Closing Price of trained Data',prediction_graph=prediction_encoded.decode('utf-8'),desc3='Closing Price prediction of next 100 days')
if __name__ == '__main__':
    app.run()

