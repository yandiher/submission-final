# ########## library

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb
import joblib

# FRONT

st.title('Jaya Jaya Institut Prediction')

marital_status = st.selectbox(
    label='marital status',
    options=[
        'single',
        'married',
        'divorced',
        'widower',
        'facto_union',
        'legally seperated'],
    label_visibility='visible'
)

application_mode = st.selectbox(
    label='application mode',
    options=[
        '2nd phase - general contingent',
        'International student (bachelor)',
        '1st phase - general contingent',
        'Over 23 years old',
        '3rd phase - general contingent',
        'Short cycle diploma holders',
        'Change of institution/course',
        'Change of course',
        'Technological specialization diploma holders',
        'Holders of other higher courses',
        'Transfer',
        '1st phase - special contingent (Madeira Island)',
        '1st phase - special contingent (Azores Island)',
        'Ordinance No. 612/93',
        'Ordinance No. 854-B/99',
        'Change of institution/course (International)',
        'Ordinance No. 533-A/99, item b2) (Different Plan)',
        'Ordinance No. 533-A/99, item b3 (Other Institution)'
    ],
    label_visibility='visible'
)

course = st.selectbox(
    label='course',
    options=[
        'Animation and Multimedia Design',
        'Tourism',
        'Communication Design',
        'Journalism and Communication',
        'Social Service (evening attendance)',
        'Management (evening attendance)',
        'Nursing',
        'Social Service',
        'Advertising and Marketing Management',
        'Basic Education',
        'Veterinary Nursing',
        'Equinculture',
        'Management',
        'Biofuel Production Technologies',
        'Informatics Engineering',
        'Agronomy',
        'Oral Hygiene'
    ],
    label_visibility='visible'
)

time_attendance = st.selectbox(
    label='time attendance',
    options=[
        'daytime',
        'evening'
    ],
    label_visibility='visible'
)

previous_qualification = st.selectbox(
    label='previous qualification',
    options=[
        'Secondary education',
        'Basic education 3rd cycle (9th/10th/11th year) or equivalent',
        'Professional higher technical course',
        '11th year of schooling - not completed',
        'Technological specialization course',
        'Higher education - degree',
        'Higher education - degree (1st cycle)',
        'Higher education - bachelor',
        'Higher education - master',
        'Other - 11th year of schooling',
        'Higher education - master (2nd cycle)',
        '10th year of schooling - not completed',
        'Frequency of higher education',
        '12th year of schooling - not completed',
        'Basic education 2nd cycle (6th/7th/8th year) or equivalent',
        'Higher education - doctorate',
        '10th year of schooling'
    ],
    label_visibility='visible'
)

mother_qualification = st.selectbox(
    label='mother qualification',
    options=[
        'Basic Education 3rd Cycle (9th/10th/11th Year) or Equivalent',
        'Secondary Education - 12th Year of Schooling or Equivalent',
        'Basic education 1st cycle (4th/5th year) or equivalent',
        'Basic Education 2nd Cycle (6th/7th/8th Year) or Equivalent',
        'Higher Education - Degree',
        'Professional higher technical course',
        'Higher Education - Bachelor',
        'Unknown',
        'Higher Education - Master',
        'Other - 11th Year of Schooling',
        'Higher education - degree (1st cycle)',
        '12th Year of Schooling - Not Completed',
        'Higher Education - Doctorate',
        'Technological specialization course',
        '7th Year (Old)',
        'Specialized higher studies course',
        '8th year of schooling',
        '10th Year of Schooling',
        'Cannot read or write',
        'Can read without having a 4th year of schooling',
        'Frequency of Higher Education',
        '11th Year of Schooling - Not Completed',
        '9th Year of Schooling - Not Completed',
        'General commerce course',
        'Higher Education - Master (2nd cycle)',
        'Technical-professional course',
        '2nd cycle of the general high school course',
        '7th year of schooling',
        'Higher Education - Doctorate (3rd cycle)'
    ],
    label_visibility='visible'
)

father_qualification = st.selectbox(
    label='father qualification',
    options=[
        'Other - 11th Year of Schooling',
        'Higher Education - Degree',
        'Basic education 1st cycle (4th/5th year) or equiv',
        'Basic Education 2nd Cycle (6th/7th/8th Year) or Equiv',
        'Secondary Education - 12th Year of Schooling or Equivalent',
        'Basic Education 3rd Cycle (9th/10th/11th Year) or Equivalent',
        'Higher Education - Master' 'Unknown ',
        'Technological specialization course',
        'Higher Education - Bachelor',
        'Higher Education - Doctorate',
        '7th Year (Old)',
        '12th Year of Schooling - Not Completed',
        'Can read without having a 4th year of schooling',
        '7th year of schooling',
        'Higher education - degree (1st cycle)',
        '10th Year of Schooling',
        'Complementary High School Course',
        'Cannot read or write ',
        'Specialized higher studies course',
        'Technical-professional course',
        '2nd year complementary high school course',
        '9th Year of Schooling - Not Completed',
        'Higher Education - Master (2nd cycle)',
        'General commerce course',
        'Professional higher technical course',
        '11th Year of Schooling - Not Completed',
        'Frequency of Higher Education',
        '8th year of schooling',
        'Complementary High School Course - not concluded',
        'Higher Education - Doctorate (3rd cycle)',
        'Supplementary Accounting and Administration',
        '2nd cycle of the general high school course',
        'General Course of Administration and Commerce',
    ],
    label_visibility='visible'
)

mother_occupation = st.selectbox(
    label='mother occupation',
    options=[
        'Personal Services, Security and Safety Workers and Sellers',
        'Intermediate Level Technicians and Professions',
        'Unskilled Workers',
        'Skilled Workers in Industry, Construction and Craftsmen',
        'Administrative staff',
        'Representatives of the Legislative Power and Executive Bodies, Directors, Directors and Executive Ma',
        'Specialists in information and communication technologies (ICT)',
        'Student',
        'Farmers and Skilled Workers in Agriculture, Fisheries and Forestry',
        'Specialists in Intellectual and Scientific Activities',
        'Other Situation',
        'Installation and Machine Operators and Assembly Workers',
        'Office workers, secretaries in general and data processing operators',
        '(blank)',
        'cleaning workers',
        'Meal preparation assistants',
        'Workers in food processing, woodworking, clothing and other industries and crafts',
        'Unskilled workers in agriculture, animal production, fisheries and forestry',
        'Technicians and professionals, of intermediate level of health',
        'Data, accounting, statistical, financial services and registry-related operators',
        'teachers',
        'Armed Forces Professions',
        'Unskilled workers in extractive industry, construction, manufacturing and transport',
        'personal service workers',
        'Intermediate level technicians from legal, social, sports, cultural and similar services',
        'Other administrative support staff',
        'Skilled construction workers and the like, except electricians',
        'Personal care workers and the like' 'sellers'
    ],
    label_visibility='visible'
)

father_occupation = st.selectbox(
    label='father occupation',
    options=[
        'Unskilled Workers',
        'Intermediate Level Technicians and Professions',
        'Skilled Workers in Industry, Construction and Craftsmen',
        'Armed Forces Professions',
        'Personal Services, Security and Safety Workers and Sellers',
        'Administrative staff',
        'Installation and Machine Operators and Assembly Workers',
        'Representatives of the Legislative Power and Executive Bodies, Directors, Directors and Executive Ma',
        'Specialists in finance, accounting, administrative organization, public and commercial relations',
        'Farmers and Skilled Workers in Agriculture, Fisheries and Forestry',
        'Student',
        'Specialists in Intellectual and Scientific Activities',
        'Other Situation',
        'Workers in food processing, woodworking, clothing and other industries and crafts',
        '(blank)',
        'Other administrative support staff',
        'Street vendors (except food) and street service providers',
        'Unskilled workers in agriculture, animal production, fisheries and forestry',
        'personal service workers',
        'Unskilled workers in extractive industry, construction, manufacturing and transport',
        'assembly workers',
        'Intermediate level science and engineering technicians and professions',
        'Meal preparation assistants',
        'Data, accounting, statistical, financial services and registry-related operators',
        'Skilled construction workers and the like, except electricians',
        'Other Armed Forces personnel',
        'Farmers, livestock keepers, fishermen, hunters and gatherers, subsistence',
        'Skilled workers in metallurgy, metalworking and similar',
        'sellers',
        'Armed Forces Sergeants',
        'Fixed plant and machine operators',
        'Vehicle drivers and mobile equipment operators',
        'Information and communication technology technicians',
        'Personal care workers and the like',
        'Skilled workers in electricity and electronics',
        'Directors of administrative and commercial services',
        'Health professionals',
        'Office workers, secretaries in general and data processing operators',
        'Hotel, catering, trade and other services directors',
        'Armed Forces Officers',
        'Protection and security services personnel',
        'teachers'
    ],
    label_visibility='visible'
)

displaced = st.selectbox(
    label='displaced',
    options=[
        'yes',
        'no'
    ],
    label_visibility='visible'
)

debtor = st.selectbox(
    label='debtor',
    options=[
        'yes',
        'no'
    ],
    label_visibility='visible'
)

tuition_fees_up_to_date = st.selectbox(
    label='tuition fees up to date',
    options=[
        'yes',
        'no'
    ],
    label_visibility='visible'
)

gender = st.selectbox(
    label='gender',
    options=[
        'male',
        'female'
    ],
    label_visibility='visible'
)

scholarship_holder = st.selectbox(
    label='scholarship holder',
    options=[
        'yes',
        'no'
    ],
    label_visibility='visible'
)

application_order = st.slider(
    label='application order',
    min_value=0.0,
    max_value=6.0,
    value=0.0,
    step=0.1
)

previous_qualification_grade = st.slider(
    label='previous qualification grade',
    min_value=95.0,
    max_value=190.0,
    value=95.0,
    step=0.1
)

admission_grade = st.slider(
    label='admission grade',
    min_value=95.0,
    max_value=190.0,
    value=95.0,
    step=0.1
)

age_at_enrollment = st.slider(
    label='age at enrollment',
    min_value=17.0,
    max_value=70.0,
    value=17.0,
    step=1.0
)

Curricular_units_1st_sem_credited = st.slider(
    label='curricular units 1st semester credited',
    min_value=0.0,
    max_value=20.0,
    value=0.0,
    step=0.05
)

Curricular_units_1st_sem_enrolled = st.slider(
    label='curricular units 1st semester enrolled',
    min_value=0.0,
    max_value=26.0,
    value=0.0,
    step=0.05
)

Curricular_units_1st_sem_evaluations = st.slider(
    label='curricular units 1st semester evaluations',
    min_value=0.0,
    max_value=45.0,
    value=0.0,
    step=0.05
)

Curricular_units_1st_sem_approved = st.slider(
    label='curricular units 1st semester approved',
    min_value=0.0,
    max_value=26.0,
    value=0.0,
    step=0.05
)

Curricular_units_1st_sem_grade = st.slider(
    label='curricular units 1st semester grade',
    min_value=0.0,
    max_value=20.0,
    value=0.0,
    step=0.05
)

Curricular_units_1st_sem_without_evaluations = st.slider(
    label='curricular units 1st semester without evaluations',
    min_value=0.0,
    max_value=12.0,
    value=0.0,
    step=0.05
)

GDP = st.slider(
    label='GDP',
    min_value=-4.5,
    max_value=4.5,
    value=0.0,
    step=0.05
)

kategorikal = np.array([
    marital_status,
    application_mode,
    course,
    time_attendance,
    previous_qualification,
    mother_qualification,
    father_qualification,
    mother_occupation,
    father_occupation,
    displaced,
    debtor,
    tuition_fees_up_to_date,
    gender,
    scholarship_holder
])
numerik = np.array([
    application_order,
    previous_qualification_grade,
    admission_grade,
    age_at_enrollment,
    Curricular_units_1st_sem_credited,
    Curricular_units_1st_sem_enrolled,
    Curricular_units_1st_sem_evaluations,
    Curricular_units_1st_sem_approved,
    Curricular_units_1st_sem_grade,
    Curricular_units_1st_sem_without_evaluations,
    GDP
])

# st.write(kategorikal)

# BACK

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier

dataset = pd.read_csv('dataset_clean.csv')
dataset.drop([dataset.keys()[0]], axis=1, inplace=True)

x = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]

onehot = OneHotEncoder(sparse_output=False)
onehot.fit(x[x.select_dtypes(include=['object']).columns.tolist()])

x_onehot = onehot.transform(x[x.select_dtypes(include=['object']).columns.tolist()].values,)
x_number = x[x.select_dtypes(include=['number']).columns.tolist()].values
x_merge = np.concatenate((x_onehot, x_number), axis=1)

scaler = MinMaxScaler()
scaler.fit(x_merge)
x_scaler = scaler.transform(x_merge)

clf = RandomForestClassifier()
clf.fit(x_scaler, y)

himpunan_kategorikal = onehot.transform([kategorikal])
himpunan = np.concatenate((himpunan_kategorikal, [numerik]), axis=1)
himpunan_scaler = scaler.transform(himpunan)

def Predict(array):
    pred = clf.predict(array)
    if pred == 0:
        return 'DROPOUT'
    else:
        return 'GRADUATE'
    
if st.button("Predict"):
    st.title(Predict(himpunan_scaler),)
else:
    st.text("")