import pandas as pd
from flask import Flask, jsonify, request, render_template
import joblib
import traceback
# load model
model = joblib.load('model.pkl')
model_columns = joblib.load('model_columns.pkl')
# app
app = Flask(__name__)

# routes
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Uncomment to support JSON
        #data = request.get_json(force=True)

        # To Support form data from HTML
        data = request.form.to_dict(flat=False)
        # convert data into dataframe
        data.update((x, y) for x, y in data.items())

        df = pd.DataFrame.from_dict(data)
        print(df.head())
        print(df.info())
        # datatype convertion
        df['AGE'] = pd.to_numeric(df['AGE'], errors='coerce')
        df['HEIGHT'] = pd.to_numeric(df['HEIGHT'], errors='coerce')
        df['WEIGHT'] = pd.to_numeric(df['WEIGHT'], errors='coerce')

        print(df.info())
        # Combine Country of residence and country of destination to form a relationship
        df['COUNTRY_REL'] = df['COUNTRY_RES'] + '_'+df['COUNTRY_DES']

        # We will create the following bins: AGE_GROUP, HEIGHT, WEIGHT

        # AGE_GROUP
        bins_age_group = [10, 20, 30, 40, 60, 70, 80]
        bin_labels_age_group = ['<20', '20-29',
                                '30-39', '40-59', '60-79', '>80']
        df['AGE_GROUP'] = pd.cut(
            df.AGE, bins_age_group, right=False, labels=bin_labels_age_group)

        # HEIGHT
        bins_height = [150, 160, 170, 180, 190, 200]
        bin_labels_height = ['<160', '160-169', '170-179', '180-189', '>190']
        df['HEIGHT_GROUP'] = pd.cut(
            df.HEIGHT, bins_height, right=False, labels=bin_labels_height)

        # WEIGHT
        bins_weight = [40, 50, 60, 70, 80, 90]
        bin_labels_weight = ['<50', '50-59', '60-69', '70-79', '>80']
        df['WEIGHT_GROUP'] = pd.cut(
            df.WEIGHT, bins_weight, right=False, labels=bin_labels_weight)

        # One hot encoding for GENDER
        one_hot_gender = pd.get_dummies(df.GENDER, prefix='GENDER')
        df = df.join(one_hot_gender)
        # One hot encoding for SMOKE_STATUS
        one_hot_smoke = pd.get_dummies(df.SMOKE_STATUS, prefix='SMOKE_STATUS')
        df = df.join(one_hot_smoke)
        # One hot encoding for AGE_GROUP
        one_hot_age_group = pd.get_dummies(df.AGE_GROUP, prefix='AGE_GROUP')
        df = df.join(one_hot_age_group)
        # One hot encoding for HEIGHT_GROUP
        one_hot_height_group = pd.get_dummies(
            df.HEIGHT_GROUP, prefix='HEIGHT_GROUP')
        df = df.join(one_hot_height_group)
        # One hot encoding for WEIGHT_GROUP
        one_hot_weight_group = pd.get_dummies(
            df.WEIGHT_GROUP, prefix='WEIGHT_GROUP')
        df = df.join(one_hot_weight_group)
        # One hot encoding for COUNTRY_REL
        one_hot_country_rel = pd.get_dummies(
            df.COUNTRY_REL, prefix='COUNTRY_REL')
        df = df.join(one_hot_country_rel)

        df = df.drop('COUNTRY_REL', axis=1)
        df = df.drop('WEIGHT_GROUP', axis=1)
        df = df.drop('HEIGHT_GROUP', axis=1)
        df = df.drop('AGE_GROUP', axis=1)
        df = df.drop('SMOKE_STATUS', axis=1)
        df = df.drop('GENDER', axis=1)
        df = df.drop('AGE', axis=1)
        df = df.drop('HEIGHT', axis=1)
        df = df.drop('WEIGHT', axis=1)
        df = df.drop('COUNTRY_RES', axis=1)
        df = df.drop('COUNTRY_DES', axis=1)

        df = df.reindex(columns=model_columns, fill_value=0)

        prediction = list(model.predict(df))

        print(prediction)
        # Uncomment to return JSON
        # return jsonify({'prediction': str(prediction)})
        return render_template('index.html', prediction_text='Recommended products : {}'.format(prediction))

    except:

        return jsonify({'trace': traceback.format_exc()})


if __name__ == '__main__':
    app.run(port=5000, debug=True)
