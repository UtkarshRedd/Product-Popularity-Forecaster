from flask import Flask, jsonify, request, render_template,url_for
from flask_cors import CORS, cross_origin
from app import config as cfg
from app.data.data_utils import get_level_wise_contribution
import json
import os
from app.logger import APP_LOGGER
from app.data.data_utils import DATA_LOADER
from app.models.sarimax.sarimax_utils import  (generate_city_wise_top_categories,
                                                generate_city_category_wise_top_products,
                                                generate_city_wise_top_categories_and_products,
                                               sarimax_prediction_configs)
app = Flask(__name__)
CORS(app)




@app.route("/popularity/api/v1.0/forecast", methods=['POST'],
           defaults={'steps': 1, 'city_name': None, 'l3_category': None, 'top_n_results': 10})
def predict(steps, city_name, top_n_results, l3_category):


    try:
        data = request.json

        configs = sarimax_prediction_configs()

        steps = data.get('steps')
        if steps:
            steps = int(steps)
        else:
            steps = 1

        top_n_results = data.get('top_n_results')
        if top_n_results:
            top_n_results = int(top_n_results)
        else:
            top_n_results = 10

        city_name = data.get('city_name')
        city_name = str(city_name).upper()

        l3_category = data.get('l3_category')

        if city_name and l3_category:

            result_dict = generate_city_category_wise_top_products(city_col=configs['city_col'],
                                                                   model_file_path=configs['model_file_path'],
                                                                   city_value=city_name, l3_col=configs['l3_col'],
                                                                   l3_value=l3_category, start_date=configs['start_date'], steps=steps,
                                                               max_model_date=configs['max_model_date'],
                                                               day_name_list=configs['day_name_list'], top_n_results=top_n_results,
                                                               category_product_contribution = configs['category_product_contribution'])

        elif city_name and not l3_category:

            result_dict = generate_city_wise_top_categories_and_products(city_col=configs['city_col'],
                                                                            model_file_path=configs['model_file_path'],
                                                                            city_value=city_name, start_date=configs['start_date'], steps=steps,
                                                               max_model_date=configs['max_model_date'],
                                                               day_name_list=configs['day_name_list'], top_n_results=top_n_results,
                                                               category_product_contribution = configs['category_product_contribution'])
        else:
            result_dict = {'error_msg': 'Invalid Request'}

        result = json.dumps(result_dict, indent=4)


    except Exception as e:

        APP_LOGGER.exception('Error In Model Prediction file '+ str(e))

    # print(result)

    #     return render_template["index.html"]
    return result


# running REST interface, port=3000 for direct test
if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=3000)