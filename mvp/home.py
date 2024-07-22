import streamlit as st

st.image('https://github.com/robchimel/MIDS210-FURever-Home-Forecast/raw/main/mvp/banner-image.jpg')

st.title('Pawpulation Forecast :dog2::black_cat:')

st.header('Our Mission')
st.markdown(""" To help animal shelters better allocate limited animal care resources by 
providing insights on how long their animals will stay once they are taken in.""")

st.header('Our Motivation')
st.markdown(""" Animal shelters have limited resources, making it critical to optimize care for animals without a 
home and under their care temporarily. Knowing how long it will take for each animal to move on from their facility 
would help animal shelters make better decisions on resource allocation. However, animal shelters struggle with accurately 
predicting this information early due to their limited knowledge about the animal at intake. With our product, we hope to 
provide animal shelters with a length-of-stay prediction report for each animal that is in their care,regardless of how recent 
their intake date is, when given a set number of attributes and features about the animal.""")

st.header('Our Solution')
st.markdown(""" For this project, we use multi-stage XGBoost modeling based on length-of-stay bucket labels to predict 
the length-of-stay bucket an animal will fall into based on limited information known about the animal during a typical 
intake process and/or as care is provided post-intake.

We use publicly available animal shelter intake and outcome data from Sonoma County as our primary dataset. 
Similar intake and outcome data from other locations like Denver Animal Protection and Austin Animal Center were also 
used as secondary datasets to evaluate generalizability. These datasets represent manually collected intake and outcome 
information of animals under care at the shelter in the past X years. In total, we are using 250,000 animal intake and 
outcome records that were manually entered, stored and maintained within these animal shelter data management systems. 
Featuring engineering was also performed on these datasets to improve model accuracy. Each animal is assigned a length-of-stay 
label based on how many days they were at the animal shelter. In addition, new features were developed from scraping keywords, 
embedding notes, and date summaries. In total, our dataset includes X descriptive attributes and 14 features of each animal record.""")

st.header('Evaluation')
st.markdown(""" To evaluate our models, we use accuracy scores, classification reports, and confusion matrices to understand the 
performance of our classification models. You can find some of our model results below:""")
##st.image('ENTER IMAGE OF MODEL PERFORMANCE')

st.header('Key Learnings & Impact')
st.markdown(""" We aimed to find the best approach for our length-of-stay prediction model. Not many academic papers had 
similar approaches to build upon, but some work has been published to solve a similar problem with similar data. 
For example, Bronzi et al developed an outcome prediction model (https://www.causeweb.org/usproc/eusrc/2020/program/8) 
using similar intake and outcome data from an Austin animal shelter. Their application intent was the same: 
help animal shelters make better decisions for animals at intake. However, their model predicts animal outcomes (i.e. adopted, transferred, died) 
and our model predicts the number of days an animal will remain at a shelter in the form of bins that represent length of time ranges. 
We believe our approach is more useful to animal shelters because it enables iterative decision making overtime. 

To develop our solution, we compared and tested models that included: Logistic Regression, Random Forest, Gradient Boosted, and XGBoost. 
XGBoost outperformed the other models, but still had relatively low performance quality for select classes. 
To address these imperfections, we incorporated multi-stage label-based model training to improve accuracy for classes that were 
very similar from one another and for classes that had low accuracy despite having unique differences from the other classes.
""")

st.header('Who We Are')
st.image('https://github.com/robchimel/MIDS210-FURever-Home-Forecast/raw/main/mvp/team-image.jpg')

st.header('Acknowledgements')
st.markdown(""" We are grateful to have the guidance and encouragement from our Capstone Advisors, 
Dr. Korin Reid and Dr. Puya H. Vahabi, of the Masters in Data Science program at School of Information, 
University of California, Berkeley.

In addition, the following subject matter experts contributed heavily during the research and/or prototype testing phases of our project:

- Monica Dangler, Director of Pima Animal Care Center (PACC)
- Kaitlyn Pappas, PACC Employee
- Katie Hutchinson, PACC Animal Placement Manager
- Melanie Sobel, Director of Denver Animal Protection
- Sarah Siskin, Adoption Manager, Humane Society of Memphis
- Joscelyne Thompson, Intake Manager, Humane Society of Memphis
""")
