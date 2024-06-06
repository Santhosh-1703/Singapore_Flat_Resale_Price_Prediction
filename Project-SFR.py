import pandas as pd
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu
import pickle
import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

# Set the page configuration with the Airbnb icon
st.set_page_config(page_title='Singapore Resale Flat Price Prediction', page_icon="sfr.jpg", layout="wide")

# Front Page Design
st.markdown("<h1 style='text-align: center; font-weight: bold; font-family: Comic Sans MS;'>Singapore Resale Flat Price Prediction</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Hello Connections! 👋 Welcome to My Project Presentation 🙏</h3>", unsafe_allow_html=True)

selected_page = option_menu(
    menu_title='',
    options=["Home","Prediction Zone","About"],
    icons=["house","trophy","patch-question"],
    default_index=1,
    orientation="horizontal",
    styles={"container": {"padding": "0!important", "background-color": "Orange","size":"cover", "width": "100"},
            "icon": {"color": "black", "font-size": "25px"},
            "nav-link": {"font-size": "15px", "text-align": "center", "margin": "-2px", "--hover-color": "#white"},
            "nav-link-selected": {"background-color": "Red"}})

if selected_page == "About":
    st.header(" :Green[Project Conclusion]")
    tab1,tab2 = st.tabs(["Features","Connect with me on"])
    with tab1:
        st.header("This Streamlit application allows users to access and analyze data from dataset.", divider='rainbow')
        st.subheader("1.    Users can select specific criteria such as Town, Street, Flat Type, Flat Model, floor sq.m, and Lease date to retrieve relevant data and analyze trends.")
        st.subheader("2.    Users can access slicers and filters to explore data. They can customize the filters based on their preferences.")
        st.subheader("3.    The analysis zone provides users with access to filters derived through Python scripting.")
        st.subheader("4.    They can explore advanced predicted values to gain deeper insights into dataset.")
    with tab2:
             # Create buttons to direct to different website
            linkedin_button = st.button("LinkedIn")
            if linkedin_button:
                st.write("[Redirect to LinkedIn Profile > (https://www.linkedin.com/in/santhosh-r-42220519b/)](https://www.linkedin.com/in/santhosh-r-42220519b/)")

            email_button = st.button("Email")
            if email_button:
                st.write("[Redirect to Gmail > santhoshsrajendran@gmail.com](santhoshsrajendran@gmail.com)")

            github_button = st.button("GitHub")
            if github_button:
                st.write("[Redirect to Github Profile > https://github.com/Santhosh-1703](https://github.com/Santhosh-1703)")

elif selected_page == "Home":
    tab1,tab2 = st.tabs(["Singapore Flats Resale Price Prediction","  Applications and Libraries Used! "])
    with tab1:
        st.write(" Singapore Flats Resale Price using a Machine Learning helps users gather valuable insights about Total Performing area, and Property details. By combining this data with information from social media, users can get a comprehensive view of their online presence and audience engagement. This approach enables data-driven decision-making and more effective content strategies.")
        st.write("[:open_book: Click here to know current Copper Price  >](https://markets.businessinsider.com/commodities/copper-price)")
        if st.button("Click here to know about this Model"):
            col1, col2 = st.columns(2)
            with col1:
                giphy_url = "https://giphy.com/embed/iRIf7MAdvOIbdxK4rR"
                giphy_iframe = f'<iframe src="{giphy_url}" width="600" height="480" frameBorder="0" class="giphy-embed" allowFullScreen></iframe>'
                giphy_link = '<p><a href="https://giphy.com/gifs/pudgypenguins-trading-chart-charts-iRIf7MAdvOIbdxK4rR">via GIPHY</a></p>'

                st.markdown(giphy_iframe + giphy_link, unsafe_allow_html=True)
            with col2:
                st.header(':white[Application info]', divider='rainbow')
                st.subheader(":star: Singapore Flats Resale Price Prediction Project involves to predict both Price based on location type & Building type")
                st.subheader(":star: To predict the Flat Price, Regression Trained Model is used. ")
                st.subheader(":star: This project aims to construct a machine learning model and implement, it as a user-friendly online application in order to provide accurate predictions about the resale values of apartments in Singapore ")
                st.subheader(":star: Resale prices are influenced by a wide variety of criteria, including location, the kind of apartment, the total square footage, and the length of the lease ")
                st.subheader(":star: The provision of customers with an expected resale price based on these criteria is one of the ways in which a predictive model may assist in the overcoming of these obstacles. ")
                
            
    with tab2:
                st.subheader("  :bulb: Python")
                st.subheader("  :bulb: Numpy")
                st.subheader("  :bulb: Pandas")
                st.subheader("  :bulb: Scikit-Learn")
                st.subheader("  :bulb: Streamlit")

elif selected_page == "Prediction Zone":
    tab1, tab2 = st.tabs(["Predict Selling Price by Flat type","Predict Selling Price by Distance "])
    with tab2:
            data = pd.read_csv(r"Singapore_flat_resale_price_file.csv")
            df = pd.DataFrame(data)
            # Define the form
            with st.form("form1"):
                col1,col2 = st.columns([10,10])
                with col1:
                    # New Data inputs from the user for predicting the resale price
                    town = st.selectbox("Select Town", df['town'].unique())
                    # Filter addresses based on the selected town
                    filtered_addresses = df[df['town'] == town]['address'].unique()
                    address = st.selectbox("Select Address", filtered_addresses)
                    flat_model = st.selectbox("Select Flat Type", df['flat_model'].unique())
                    flat_type = st.selectbox("Select Flat Type", df['flat_type'].unique())
                    month = st.selectbox("Select Transaction Month", df['month'].unique())
                    st.info("Info: Getting MRT (Mass Rapid Transit System) Railway Transportation for each Town So that we can calculate distance form coordinates.")
                    st.info("Info: Getting coordinates of each HDB(Housing and Developing Board) Resale flat in order to conduct the distance from MRT stations. So, that we can take out the distance of flats from MRT stations (Mass Rapid Transit System) and also from CBD (Central Business District).")
                with col2:
                    # Filter the dataframe based on the selected address
                    filtered_df = df[df['address'] == address]
                    # Extract CBD distance and Min distance to MRT
                    if not filtered_df.empty:
                        cbd_dist_min = filtered_df['cbd_dist'].min()
                        cbd_dist_max = filtered_df['cbd_dist'].max()
                        min_dist_mrt_min = filtered_df['min_dist_mrt'].min()
                        min_dist_mrt_max = filtered_df['min_dist_mrt'].max()
                    else:
                        cbd_dist_min = 593  # Default min value for slider
                        cbd_dist_max = 20224  # Default max value for slider
                        min_dist_mrt_min = 0  # Default min value for slider
                        min_dist_mrt_max = 19227  # Default max value for slider
                    #Slider for CBD Distance
                    cbd_dist = st.slider("Select CBD Distance", min_value=0, max_value=int(cbd_dist_max), value=int(cbd_dist_min))
        
                    # Slider for Min Distance to MRT
                    min_dist_mrt = st.slider("Select Min Distance to MRT", min_value=0, max_value=int(min_dist_mrt_max), value=int(min_dist_mrt_min))
                    floor_area_sqm = st.slider("Select Floor Area (Sq. m)", min_value=32, max_value=280, value=80)
                    lease_remain_years = st.slider("Select Lease Remaining (years)", min_value=41, max_value=97, value=60)
                    storey_median = st.slider("Select Storey Range Median", min_value=0, max_value=50, value=25)

                    # Submit Button for PREDICT RESALE PRICE
                    submit_button = st.form_submit_button(label="PREDICT RESALE PRICE")

            if submit_button:
                user_input = {
                    'cbd_dist': cbd_dist,
                    'min_dist_mrt': min_dist_mrt,
                    'floor_area_sqm': floor_area_sqm,
                    'lease_remain_years': lease_remain_years,
                    'storey_median': storey_median
                }

            new_sample = np.array([[cbd_dist, min_dist_mrt, np.log(floor_area_sqm), lease_remain_years, np.log(storey_median)]])
            with open(r"scaler.pkl", 'rb') as f:
                    scaler_loaded = pickle.load(f)
            with open(r"modelxgb.pkl", 'rb') as file:
                    loaded_model = pickle.load(file)
            new_sample = scaler_loaded.transform(new_sample[:, :5])
            new_pred =   loaded_model.predict(new_sample)[0]
            with col2:
                st.write('## :green[Predicted Resale Price: 💲] ', np.exp(new_pred))
    
    with tab1:
            data = pd.read_csv(r"C:\\Users\\SANTHOSH RAJENDRAN\\Desktop\\GUVI Python\\Project-Singaporeflatresale\\SRF_Processed.csv")
            df1 = pd.DataFrame(data)
            with st.form("form2"):
                    col1,col2=st.columns(2)
                    with col1:
                        # Load the pickled encoders
                        with open(r"C:/Users/SANTHOSH RAJENDRAN/Desktop/GUVI Python/Project-Singaporeflatresale/categorical_encoders.pickle", 'rb') as f:
                            encoders = pickle.load(f)

                        # Load the pickled model
                        with open(r"C:/Users/SANTHOSH RAJENDRAN/Desktop/GUVI Python/Project-Singaporeflatresale/modelxgbsfr.pkl", 'rb') as file:
                            model = pickle.load(file)

                        # Define a function to preprocess input features
                        def preprocess_input(town, block, street_name, flat_type, flat_model, storey_min, storey_max, transaction_month, transaction_year, lease_commence_date, end_of_lease, floor_area_sqm):
                            # Encode categorical features
                            town_encoded = encoders['town'].transform([town])[0]
                            flat_type_encoded = encoders['flat_type'].transform([flat_type])[0]
                            block_encoded = encoders['block'].transform([block])[0]
                            street_name_encoded = encoders['street_name'].transform([street_name])[0]
                            flat_model_encoded = encoders['flat_model'].transform([flat_model])[0]

                            # Create a dataframe with the preprocessed features
                            data = data = pd.DataFrame({
                                        'transaction_month': [transaction_month],
                                        'town': [town_encoded],
                                        'flat_type': [flat_type_encoded],
                                        'block': [block_encoded],
                                        'street_name': [street_name_encoded],
                                        'storey_min': [storey_min],
                                        'storey_max': [storey_max],
                                        'floor_area_sqm': [floor_area_sqm],
                                        'flat_model': [flat_model_encoded],
                                        'lease_commence_date': [lease_commence_date],
                                        'end_of_lease': [end_of_lease],
                                        'transaction_year': [transaction_year]
                                    })
                            return data

                        transaction_month = st.slider("Select Item Month", min_value=1, max_value=12, value=1)
                        town = st.selectbox("Select Town", df1['town'].unique())
                        flat_type = st.selectbox("Choose Flat Type", df1['flat_type'].unique())
                        block = st.selectbox("Choose Block", df['block'].unique())
                        street_name = st.selectbox("Choose Street name", df['street_name'].unique())
                        storey_min = st.number_input("Enter Storey min", min_value=1, max_value=49, value=1)
                        storey_max = st.number_input("Enter Storey max", min_value=1, max_value=51, value=1)
                        
                    
                    with col2:
                        floor_area = st.slider("Select floor sq. m", min_value=29, max_value=306, value=1)
                        floor_area_sqm = np.log(floor_area)
                        flat_model = st.selectbox("Choose Flat Model", df['flat_model'].unique())
                        lease_commence_date = st.number_input("Select Lease Commence Year which you want", min_value=1967, max_value=3000, value=2024)
                        end_of_lease = st.number_input("Select Lease End year which you want", min_value=2065, max_value=3000)
                        transaction_year = st.number_input("Select transaction Year which you want", min_value=1990, max_value=3000, value=2024)
                        
                        # Submit Button for PREDICT RESALE PRICE
                        submit_button1 = st.form_submit_button(label="PREDICT RESALE PRICE")
                    
                        if submit_button1:
                            # Preprocess the input features
                            input_data = preprocess_input(town, block, street_name, flat_type, flat_model, storey_min, storey_max, transaction_month, transaction_year, lease_commence_date, end_of_lease, floor_area_sqm)
                            
                            # Predict the resale price using the model
                            resale_price_log = model.predict(input_data)
                            resale_price = np.exp(resale_price_log)
                            
                            # Display the predicted resale price
                            st.write(f'## :green[Predicted Resale Price: 💲]{resale_price[0]}')
                 
