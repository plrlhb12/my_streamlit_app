import streamlit as st
import pandas as pd 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# create container
header = st.container()
dataset = st.container()
features = st.container()
model_raining = st.container()

# make some customization on displaying
st.markdown(
	"""
	<style>
	.main{
	background-color: #F5F5F5;
	}
	</style>
	""",
	unsafe_allow_html=True
	)

# to allow cache to allow streamlit to use the previously saved results instead of running from the beginning
@st.cache
def get_data(filename):
	taxi_data = pd.read_csv(filename)
	return taxi_data

# write some texts inside a container
with header:
	st.title("Welcome to my data science project!")
	st.text("In this project I look into the transactions of taxis in NYC...")

with dataset:
	st.header("NYC taxi dataset")
	st.text("I found this dataset on https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page")

	taxi_data = get_data("data/yellow_tripdata_2019-01.csv")
	st.write(taxi_data.head())

	# viz the pickup locations's stats
	st.subheader("Pickup location")
	pulocation_dist = pd.DataFrame(taxi_data['PULocationID'].value_counts().head(50))
	st.bar_chart(pulocation_dist)

with features:
	st.header("The features I created")
	st.text("In this project I look into the transactions of taxis in NYC...")

	# make a list using the function of markdown
	st.markdown("* **first feature:** I created this feature because of this... I calclauted it using ...")
	st.markdown("* **second feature:** I created this feature because of this... I calclauted it using ...")


with model_raining:
	st.header("Time to train the model")
	st.text("In this project I look into the transactions of taxis in NYC...")

	# create 2 columns in this container, the left one is called sel_col and the right is called disp_col
	sel_col, disp_col = st.columns(2)

	# set the selection column to get the users' input and assign to an parameter oject for ML algorithsm
	# create slider using col_object.slider()
	# also get the user's input: save whatever the uses choses as max_depth
	max_depth = sel_col.slider("What should be the max_depth of the model", 
		min_value=10, max_value=100, value=20, step=10)

	# index=0 means the default value is that of the first item in the options list
	n_estimators = sel_col.selectbox("How many trees sould there be?", options=[100,200,300,"No limit"], index=0)

	# first provide a list of features for reference, it will automatically generate scroll bar
	# also can use sel_col.table() instead of sel_col.write()
	sel_col.text("Here is a list of features in my data:")
	sel_col.write(taxi_data.columns)
	# the second string is the default text showing up in the text_input box
	input_feature = sel_col.text_input("Which feature should be used as the input feature", "PULocationID")

	if n_estimators == 'no limit':
		# set up random forest and take the input from selection column as algorithsm parameters
		regr = RandomForestRegressor(max_depth=max_depth)
	else:
		regr = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)

	X = taxi_data[[input_feature]]
	y = taxi_data[["trip_distance"]]
	regr.fit(X,y)
	prediction=regr.predict(y)

	# set the display column, it can be called repeatly to generate multiple lines and multiple execution
	disp_col.subheader("Mean absolute error of the model is:")
	disp_col.write(mean_absolute_error(y, prediction))
	disp_col.subheader("Mean squared error of the model is:")
	disp_col.write(mean_squared_error(y, prediction))
	disp_col.subheader("Mean squared error of the model is:")
	disp_col.write(r2_score(y, prediction))






