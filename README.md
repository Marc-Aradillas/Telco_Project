Telco Project
=============================
<!--Created Anchor links to navigate read me better-->

- [Project Description](#project-description)
- [Project Goal](#project-goal)
- [Initial Thoughts](#Initial-Thoughts)
- [Plan](#Plan)
- [Data Dictionary](#data-dictionary)
- [Steps to Reproduce](#steps-to-reporduce) 
- [Conclusions](#conclusions)
	- [Takeaway and Key Findings](#takeaways-and-key-findings)
	- [Reccomendations](#recommendations)
	- [Next Steps](#next-steps)

----------------------------------

# Project-Description : Analyzing Customer Churn Factors for Predictive Insights

**The Telco Telecommunication company provides phone, internet, streaming and other add-on services to their customers. This project will involve analysis of the various elements of customer churns to determine if they increase or decrease the probability of customer churn  **

## Project Goal

* Identify drivers for churn of Telco customers
* Utilze drivers to develop a Machine Learning model to classify churn as a customer ending their contract or not ending their contract with Telco.
* The details of could provide further insight on which customer elements contribute to or detract from a customer churning.

### Initial Thoughts

My initial hypothesis is that the drive of churn may be impacted by customers not opting for online security as part of their contract services.

### The Plan

* **Acquire data from Codeup MySQL Database**
  
* **Prepare data**
    * Feature Engineer columns from existing data
        * online_security
        * online_backup
        * device_protection
        * tech_support
        * streaming_tv
        * streaming_movies
        * tech_support
      
* **Explore data in search of impactful drivers of churn**
    * Answer the following initial questions,
        * Is Churn independent from online Security?
        * Is Churn independent from internet service types?
        * Is there a difference in churn based on monthly charges? Total charges?
        * Does having more than one add-on affect churn?
        * Does the contract type affect churn?
          
* **Develop a Model to predict if a customer will churn**
    * Use drivers identified in explore to help build predictive models of different types
    * Evaluate models on train and validate data
    * Select the best model based on highest accuracy
    * Evaluate the best performing model on test data
      
* **Draw conclusions**


## Data Dictionary

|**Feature**|**Values**|**Definition**|
|:--------------------:|:---------------------------:|:-------------------------------------------------------- |
| *customer_id*           | Alpha-numeric               | Unique ID for each customer                              |
| *gender*                | Female/Male                 | Gender of customer                                       |
| *senior_citizen*        | True=1/False=0              | Whether customer is a senior citizen or not              |
| *partner*               | True=1/False=0              | True=1/False=0, whether customer has a partner or not    |
| *dependents*            | True=1/False=0              | True=1/False=0, whether customer has dependents or not   |
| *phone_service*         | True=1/False=0              | True=1/False=0, whether customer has phone service or not|
| *multiple_lines*        | Yes/No/No phone service     | Whether customer has multiple lines or not               |
| *internet_service_type* | None/DSL/Fiber Optic        | Which internet service customer has                      |
| *online_security*       | Yes/No/No internet service  | Whether customer has online_security                     |
| *online_backup*         | Yes/No/No internet service  | Whether customer has online_backup                       |
| *device_protection*     | Yes/No/No internet service  | Whether customer has device_protection                   |
| *tech_support*          | Yes/No/No internet service  | Whether customer has tech_support                        |
| *streaming_tv*          | Yes/No/No internet service  | Whether customer has streaming_tv                        |
| *streaming_movies*      | Yes/No/No internet service  | Whether customer has streaming_movies                    |
| *contract_type*         | 3 options                   | Month-to-Month/One-year/Two-year, term of contract       |
| *payment_type*          | 4 options (2 auto)          | Customer payment method                                  |
| *paperless_billing*     | True=1/False=0              | Whether a customer has paperless billing enabled         |
| *monthly_charges*       | Numeric USD                 | Amount customer is charged monthly                       |
| *total_charges*         | Numeric USD                 | Total amount customer has been charged                   |
| *tenure*                | Numeric                     | Number of months customer has stayed                     |
| *churn* (target)        | True=1/False=0              | Whether or not the customer has churned                  |
| *Additional Features*   | True=1/False=0              | Encoded values for categorical data                      |


## Steps to Reproduce

1) Clone this repo following the code link at the top
2) If you have granted access to the Codeup MySQL DB:
   - Save **env.py** in the repo with `user`, `password`, and `host` variables and add to .gitignore file
   - Run jupyter notebook or VS Code file
3) If you don't have granted access:
   - Request access from Codeup staff
   - Return to step 2

## Conclusions

### Takeaways and Key Findings
<!-- EX:
* Payment type was found to be a driver of churn
    - Electronic check being the most common among churn
* Fiber optic internet being a driver of churn was surprising considering it has faster internet speeds
* Paperless billing was found to be a driver of churn
    - Having it enabled was the most common among churn
* Monthly charges was found to be a driver of churn, higher charges for churn-->

### Recommendations

<!-- EX:
* Check with the finance department to figure out if there are issues with electronic check
* Check with internet department to see if there are frequent issues with Fiber optic internet-->

### Next Steps

<!-- EX:
* Given more time I could check what is causing the high monthly charges that lead to customers churning-->
