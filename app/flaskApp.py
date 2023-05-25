from models.sarimax.sarimax_utils import (generate_city_wise_top_categories,
                                              generate_city_category_wise_top_products,
                                              generate_city_wise_top_categories_and_products,
                                              generateProductPopularity
                                              )

from flask import Flask, render_template, request, url_for
import config as cfg

app = Flask(__name__)


START_DATE ='2021-10-19'
MAX_MODEL_DATE = '2021-10-18'
DAY_NAME_LIST = ['Monday','Thursday']
CONTRIBUTION_DIR = 'contributions/'
L3_COL = 'L3'
MODEL_TYPES = {'revenue': 'revenue_models', 'quantity': 'quantity_models'}
WEIGHTS = {'revenue': 0.7, 'quantity': 0.3}
MODEL ='sarimax'
MODEL_DIR = 'models'
MODEL_FILE_PATH = f'{MODEL_DIR}/{MODEL}/trained_models'



@app.route('/')
def index():
    return render_template('index.html')


@app.route('/get_top_categories/')
def get_top_categories():
    return render_template('top_categories_request.html')


@app.route('/get_top_products/')
def get_top_products():
    return render_template('top_products_request.html')


@app.route('/get_top_categories_and_products/')
def get_top_categories_and_products():
    return render_template('top_categories_and_products_request.html')


@app.route('/get_global_product_popularity/')
def get_global_product_popularity():
    return render_template('global_products_popularity_request.html')


@app.route('/display_top_categories')
def display_top_categories():
    city_name = request.args.get('city_name').upper()
    bottom = request.args.get('get bottom categories')
    # bottom = bool(int(bottom))
    if bottom == 'Yes':
        bottom = True
    else:
        bottom = False

    # print('bottom value', bottom,type(bottom))
    # l3_category = request.args.get('l3_category')

    # configs = sarimax_prediction_configs()

    messages = generate_city_wise_top_categories(model_file_path=MODEL_FILE_PATH, city_value=city_name,
                                                 start_date=START_DATE, steps=1,
                                                 max_model_date=MAX_MODEL_DATE,
                                                 day_name_list=DAY_NAME_LIST, top_n_results=10,
                                                 model_types=MODEL_TYPES, bottom=bottom, weights=WEIGHTS)

    return render_template('display_top_categories.html', city_name=city_name, messages=messages)


@app.route('/display_top_products')
def display_top_products():
    city_name = request.args.get('city_name').upper()
    l3_category = request.args.get('l3_category').capitalize()
    bottom = request.args.get('get bottom products')
    if bottom == 'Yes':
        bottom = True
    else:
        bottom = False
    # print(bottom)

    # configs = sarimax_prediction_configs()

    messages = generate_city_category_wise_top_products(model_file_path=MODEL_FILE_PATH, city_value=city_name,
                                                        l3_col=L3_COL, l3_value=l3_category,
                                                        start_date=START_DATE, steps=1,
                                                        max_model_date=MAX_MODEL_DATE,
                                                        day_name_list=DAY_NAME_LIST, top_n_results=10,
                                                        contribution_file_path=CONTRIBUTION_DIR,
                                                        model_types=MODEL_TYPES, weights=WEIGHTS, bottom=bottom)
    # print(messages)
    # total_sum = sum(messages.values())
    return render_template('display_top_products.html', city_name=city_name, l3_category=l3_category, messages=messages)


@app.route('/display_top_categories_and_products')
def display_top_categories_and_products():
    city_name = request.args.get('city_name').upper()



    bottom = request.args.get('get bottom categories and products')
    if bottom == 'Yes':
        bottom = True
    else:
        bottom = False

    messages = generate_city_wise_top_categories_and_products(model_file_path=MODEL_FILE_PATH, city_value=city_name,
                                                              start_date=START_DATE, steps=1,
                                                              max_model_date=MAX_MODEL_DATE,
                                                              day_name_list=DAY_NAME_LIST, top_n_results=10,
                                                              contribution_file_path=CONTRIBUTION_DIR,
                                                              l3_col=L3_COL,
                                                              model_types=MODEL_TYPES, bottom=bottom,
                                                              weights=WEIGHTS)
    # print(messages)

    return render_template('display_top_categories_and_products.html', city_name=city_name, messages=messages)


@app.route('/display_global_product_popularity')
def display_global_product_popularity():
    domain_id = 'grocery'
    city_name = None
    POPULARITY_DIR = 'saved_popularity/'

    product_skuid_list = [int(_) for _ in request.args.get('product_sku_id').split(',')]

    messages = generateProductPopularity(domain_id=domain_id, product_skuid_list=product_skuid_list,
                                         city_name=city_name, popularity_dir=POPULARITY_DIR,
                                         w_q=0.7, w_r=0.3)

    return render_template('display_global_products_popularity.html', messages=messages)

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0', port=5002)
    # display()
