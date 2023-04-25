[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-c66648af7eb3fe8bc4f294546bfd86ef473780cde1dea487d3c4ff354943c9ae.svg)](https://classroom.github.com/online_ide?assignment_repo_id=10479380&assignment_repo_type=AssignmentRepo)
# Detecting Sarcasm in Social Media Comments

## Executive Summary
Our project aims to detect sarcasm in social media comments using Reddit data from 2009 to 2016. The rise of social media has led to an increase in online communication, which has led to a rise in the use of sarcasm. According to Wong (2022), the use of irony can make it difficult for social networks to determine the intention of a user, while also allowing for some degree of denial.**<sup>[1]</sup>** For example, racist comments written with irony on online platforms may go undetected and cause disharmony. Moreover, detecting sarcasm in social media comments can have various benefits, including capturing the intention of the user and facilitating online advertisement. For example, sarcastic comments that would otherwise go unnoticed could be detected, enabling the capture of user intention and thus, avoiding the presentation of irrelevant ads. As an instance, if a user leaves a sarcastic comment on a certain automobile feature online, detecting the sarcasm can allow the marketing agents to avoid presenting ads about automobile features that the user does not like. However, accordiong to Peters (2018), detecting sarcasm in online communication is difficult due to the lack of non-verbal cues.**<sup>[2]</sup>** 

Our project aims to solve this problem by using machine learning techniques to analyze Reddit data and identify patterns that indicate sarcasm. By accurately detecting sarcasm, we can improve the effectiveness of sentiment analysis and improve our understanding of online communication. Our recommendations include developing a machine learning model that can accurately detect sarcasm in social media comments. This model can be used by social media platforms to improve their sentiment analysis capabilities and better understand user behavior. This project benefits social media platforms, businesses, and researchers who are interested in understanding online communication and sentiment analysis, especially those who wish to maintain harmony on online platform and those who wish to devise a new online marketing strategy. 

**Team Members:**

- Frankle Muchahary
- Jayke Ratliff
- Omkar Patade
- Shreyan Datta Chakraborty
- Roe Djer Tan 

## Statement of Scope

For this project, we will be focusing on a `specific sub-type of sentiment analysis` which is `detecting the presence of sarcasm in social media comments`. The social media platform that we will be focusing on for our project is `Reddit` mainly due to the availability and easy access of large amounts of anonymous data. Our attempt is to build a binary classification model that can utilize both text data and other numeric data to predict the presence of sarcasm in a tweet. Our goal is build a model that can generalize well universally, however, it is important to note that the observations or findings from our project and the predictive models built for our project might not necessarily translate well to all social media platform comments. 

This exercise is done based on the data gathered from Reddit (obtained through a Kaggle dataset). In addition to the Kaggle Dataset we have also gathered supporting data on our own by scraping user-related information from Reddit. The comments that we have used for our project are limited to 3 very popular subreddits on Reddit - `AskReddit`, `politics`, and `worldnews`. Hence, we would also like to note that the observations made from our project might not always necessarily translate well to some topics like `sports`, `science`, or `technology`.  For this project, we will be trying to develop a way to predict sarcastic comments based on the `comments` itself and the related `parent comment` which will further be supported with numeric metrics like `upvotes`, `downvotes`, `karma`, etc. 

Finally, the project objectives are: 
* `Topic Analysis` (Unsupervised Machine Learning Problem): Identifying which topics are popular in sarcastic comments compared to the ones that are popular for non-sarcastic ones. 
* `Binary Classification` (Supervised Machine Learning Problem): Building a binary classification model for predicting whether a comment is sarcastic or not.

Unit of Analysis: A comment in Reddit <br>

## Project Schedule

We have divided the project into 5 different phases in order to be able to tackle the different pieces of the project in a much more structured manner. These phases are the following: 
* `Phase 1:` This phase involves definining the problem statement and brainstorming for various ideas, data sources, and access techiniques
* `Pahse 2:` In this phase we will be extracting data from different sources and cleaning the obtained data
* `Phase 3:` We will begin with the data transformation and reduction steps in this phase of the project
* `Phase 4:` Exploratory Data Analysis will be carried out on the cleaned and transformed datasets in this phase of the project. We will try to generate insights and identify which variables or metrics could help us in the predictive modeling phase of the project. We will be completing all the tasks till phase 4 for Project Deliverable 1.
* `Phase 5:` In this phase we will mostly be focusing on developing predictive models for our task i.e. Detecting Sarcasm in Reddit Comments and generating insights based on the model outputs. This phase will be carried out for Project Deliverable 2.

A detailed description of the tasks for each phase and the respective task owners have been shown in the timeline chart below. 

The snapshot of our project timeline has been shown below. However, the latest timeline could change due to various circumstances and the latest project schedule/timeline can be accessed through this [link](https://ostatemailokstate-my.sharepoint.com/:x:/g/personal/frankle_muchahary_okstate_edu/EYBcPhLyA1xGmlLA0gCi06oBLNKqCMFBKHXVvaekHbJZyA?e=ZilqS3).

<br>

![](assets/timeline_chart_1.png)

<br>

## Data Preparation

The data preparation process for this project includes obtaining data from `Kaggle`, extracting additional user-related information through scraping, cleaning the data, carrying out preprocessing for the text columns in the dataset, and finally consolidating the datasets obtained from different sources or preprocessing methods into a combined master dataset that can be used down the road for predictive modeling purposes. A majority of the data preparation was carried out using python and the following python libraries were used for the data preparation process: `asyncio`, `beautifulsoup4`, `dateutil`, `multiprocessing`, `nltk`, `pandas`, `requests`, and `scikit-learn`

<br>

The data prepration process for this project begins with gathering data from two sources. The primary focus of the project is to detect the presence of sarcasm in various texts. It is a specific sub-area of sentiment analysis. For this purpose, we obtained a `dataset from Kaggle containing Reddit comments` along with additional continuous and categorical features like upvotes, downvotes, usernames, subreddit names, etc. The dataset has also already been labeled as either `Sarcasm` or `Not Sarcasm`. In addition to this, we also obtained additional information about the users and their Reddit usage history by `scraping user-related information from Reddit` using the usernames that were available in the Kaggle dataset.  

<br>

The data access process was followed by cleaning and extraction of relevant data from both the scraped data and the `text columns (comments and parent comments)` available in the Kaggle dataset. The cleaning process included the extraction of `relevant user information like karma points, awards, etc.` from the scraped HTML data. The comments data also went through a cleaning process that involved the usual steps followed for cleaning text data like lowercase conversion, removing punctuations and numeric characters, lemmatization, etc.

<br>

The cleaning step was then followed by `tokenization` and `vectorization` of the text comments data to convert them into a format that is usable for predictive modeling. This vectorization step produces a high-dimension dataset with a large number of columns. Hence, in order to reduce the dataset size, a `data reduction step using PCA` was also utilized to convert the dataset into a more manageable size.

<br>

Finally, the data preparation process ends with a consolidation step that involved combining together all the datasets obtained from the different sources and outputs from different preprocessing steps into a combined master dataset that can be used readily for further steps of the project. The datasets and code files used for each step are available below.

<br>

**Datasets:**
* [base-data-sarcasm.csv](data/base_data/base-data-sarcasm.csv): Base Dataset containing reddit comments and sarcasm labels obtained from *Kaggle*
* [user_info.csv](data/user_data/user_info.csv): Scraped and Cleaned User Info Data obtained from *Reddit*
* [final_master_data.csv](data/final_data/final_master_data.csv): Final Merged Master data to be used for Predictive Analytics. Contains all comment related metrics, vectorized and reduced text data,  user related metrics, and also the original comments.

**Code Files:**
* [scrape_clean_user_info.py](code/scrape_user_info/scrape_clean_user_info.py): Python code for scraping and cleaning user information from Reddit using the usernames available in the base dataset
* [text_cleaning_vectorization.py](code/text_data_prep/text_cleaning_vectorization.py): Python code for cleaning the base data and converting the text data numeric vectors
* [data_cleaning_transformation_and_reduction_comments.py](code/data_cleaning_transformation_and_reduction_comments.py): Python code for cleaning the vectorized data for the comments file and applying PCA
* [data_cleaning_transformation_and_reduction_parent_comments.py](code/data_cleaning_transformation_and_reduction_parent_comments.py): Python code for cleaning the vectorized data for the parent comments file and applying PCA
* [combine_two_datasets_along_with_the_target_variable.py](code/combine_two_datasets_along_with_the_target_variable.py): Python code for creating the final data set along with the target variable and joining the two datasets on which PCA has been done
* [visualization_and_eda.R](code/visualization_and_eda.R): R code used for performing *Exploratory Data Analysis* and creating visualizations for the exploratory analysis using ggplot.

<br>

### Data Access
The data access step for this project involves obtaining dataset from two different sources: a pre-curated and pre-labeled dataset obtained from Kaggle and scraped user-related information obtained by scraping user pages on Reddit. Both these steps have been explained in detail below.

<br>

**Source 1: Pre-curated and pre-labeled sarcasm dataset obtained from Kaggle**   
A dataset containing 1.3 million Reddit Comments and other details related to the comments like number of upvotes, number of downvotes, name of the author, etc. was published on arXiv by Khodak et al. as a part of their article titled *“A Large Self-Annotated Corpus for Sarcasm”* to enable research on sarcasm detection and for training machine learning models that are capable of sarcasm detection.**<sup>[3]</sup>**  This dataset was also made available on Kaggle by the user Dan Ofer as a part of Kaggle Datasets.**<sup>[4]</sup>**  This dataset is a very rich dataset containing of a large number of reddit comments and well validated target labels generated by researchers which makes it a very reliable data source free from discrepancies. Additionally, Reddit is popular for being an anonymous social media platform which makes it a good source for obtaining user related information because the chances of breaching user privacy is lower due to the aspect of anonymity. Finally, the researchers that curated the data have also attempted to make the dataset relatively well balanced in terms of number of sarcastic and non-sarcastic comments. All these factors combined together make this a very solid dataset for analyzing sarcasm.    

Out of the original 1.3 million comments we have used a smaller sample consisting of around 7% of the dataset for use in our project to ensure a more manageable dataset size. We did this by considering only the top 3 subreddits available in the dataset based on the number of comments. These subreddits are `AskReddit`, `politics`, and `worldnews`. The total number of comments after sampling totaled to rougly 75K comments and the total number of users available to us in this sampled dataset is around 45K.   

The following variables are available in the dataset: *sarcasm label, reddit comment made by the author (text column), author of the comment (username), subreddit name, comment score, number of upvotes on the comment, number of downvotes on the comment, comment date, comment full timestamp, parent comment (text column), and a unique id for each comment.*  

<br>

`Basic Summary of the Dataset Obtained from Kaggle: `  

![](assets/data_access_1.png)

<br>

**Source 2: Scraped User Information obtained from Reddit**  
In order to consolidate the dataset obtained from the first source explained above, we have also scraped additional information mainly related to Reddit usage history for all users. This step uses the `author of comment (reddit username)` available in the dataset obtained from the first source to obtain the additional user information. 
In order to begin scraping, we first construct the URL that needs to be scraped for each user. The username is passed as a parameter to the following URL:  

https://old.reddit.com/user/ThisIsNotKimJongUn/gilded/  

In the above URL, the term between `user` and `gilded` which is `ThisIsNotKimJongUn` in this particular example is the Reddit username available to us in the author column of the dataset obtained from the first source. Hence, we loop through the dataset and construct this URL for each user before beginning the scraping process.
The URL mentioned above provides us with the following information about the user:
* `Post Karma` (Karma is the points system on Reddit. Similar to likes on other social media platforms) the user has obtained through posts created by them on Reddit.
* `Comments Karma` the user has obtained through comments they have made on Reddit.
* `Joining Date`
* `Gilded Posts or Comments`. A comment or post is considered gilded if the user receives some awards for their comments or posts. Usually comments that are considered extremely good by other users are the ones that receive awards. These comments usually tend to be extremely funny or highly sarcastic. Hence, these can help in potentially detecting sarcasm.   

A snapshot of the webpage and the content that we are trying to extract from the webpage has been highlighted in green in the image shown below.   

<img src = 'assets/data_access_2.png' width=700, height=300>

<br>

The two images shown below highlights the HTML elements corresponding to the content that we are trying to scrape from the webpage. These are the HTML elements that we will be extracting using the library `beautifulsoup4`.

<img src = 'assets/data_access_3.png' width=700, height=400>

<br>

<img src = 'assets/data_access_4.png' width=700, height=400>

<br>

Hence, to conclude, the scraping process involves three key steps which are as follows:
1. Construct URLs like the one shown above for all 45K users available in the dataset.

2. Fetch the HTML content of the webpage using the `requests library` in python. This process is being carried out in an asynchronous manner using the `asyncio libary` because we have around 45K webpages to fetch the HTML content from which takes a lot of time. Hence, by making this process as an asynchronous process, we are able to scrape the HTML content for 1000 users in parallel. This helps in speeding up the process by a significant amount. The following snippets of code are used for carrying out this process. 

```python
try:
    headers = {
        'User-Agent':UA_LIST[randint(0,4)]
    }

    rq_result = rq.get(
        url,
        headers = headers,
        timeout = 10
    )

    return rq_result

except Exception as e:
    err = (url, str(e))
    ERRORS_LIST.append(err)

``` 

```python
for single_batch_url_list in tqdm(url_list_chunks[0:dry_run_index1]):
    event_loop = asyncio.get_event_loop()
    
    # asynchronous call to the wrapper function
    Executor = ThreadPoolExecutor(max_workers=len(single_batch_url_list))
    tasks = [event_loop.run_in_executor(Executor, async_request_extract_html_wrapper, url) for url in single_batch_url_list]

    #single_batch_result = await asyncio.gather(*tasks)
    single_batch_result = event_loop.run_until_complete(asyncio.gather(*tasks))

    all_results_list.extend(single_batch_result)
```

3. Finally, in the third step of the process we use the `beautifulsoup4 libary` in python to obtain the relevant HTML elements shown in the images above by accessing through *class* and *id* attributes. Based on this, we have extracted the required user information for further use in our analysis. Sample code snippets used for this process have been shown below. 

```python
html_result = BeautifulSoup(rq_result.content, "html.parser")

### extract post karma obtained by user
extracted_elem = html_result.find(
    'span', {'class':'karma'}
)

### extract number of comments/posts the user received awards for - gilded posts and comments
all_gilded_posts = html_result.find_all(
    'div', attrs={'class':['gilded', 'thing']}
)
```

<br>

### Data Cleaning

The data cleaning process in this project consisted of multiple segregated sections that were applied at various points and on different datasets used in the project. 

**Text Data Preprocessing (For `comment` and `parent_comment` columns)**

In order to be able to use the text for exploratory analysis and predictive analysis down the road, we need to clean up the text data and convert it into a standard format. This will help us in extracting meaningful insights from our analysis and avoid drawing faulty conclusions. The text cleaning process involves the following different steps:

1. **Lowercase Conversion:** We have converted all the comments and parent comments into lower case to ensure a consistent format and to ensure that words like `happy` and `Happy` are not being considered as different words just due to their casing conventions.  
The following code snippet was used for performing this analysis:
```python
    # convert to lowercase
    self.df['cleaned_'+self.text_column_name] = self.df[self.text_column_name].str.lower()
```

2. **Removing Special Characters and Numbers from Text:** Following the lowercase conversion step mentioned above, the next step of the process involves removing special characters like `#, &, @` etc, and `numeric characters` to keep only the white spaces and text characters. This has been done to ensure that we consider only meaningful characters in our comments and parent comments and not considering a special character that has a high occurence as an individual entity later on while vectorizing the text.
The following code snippet was used for performing this analysis:
```python
    # remove numeric values - keep text only
    self.df['cleaned_'+self.text_column_name] = self.df['cleaned_'+self.text_column_name].str.replace("[^a-zA-Z']", " ", regex=True)
```

3. **Remove Stopwords:** Stopword removal is an important step while performing text analysis. Stopwords are usually words like `the`, `a`, `an`, `and`, `in`, etc. These words do not usually carry much meaning or add a lot of information to the overall context of the sentence. These words do not help us a lot in understanding the meaning of the sentence and act more like noise. Additionally, these words will also have high occurences and will overshadow other more meaningful words. Hence, removing these words will help improve the quality of our analysis.  
The following code snippet was used for performing this analysis:
```python
    def remove_stopwords(self, txt):
        '''
        helper method for removing stopwords
        '''
        txt = txt.split(' ')

        txt = [i for i in txt if i not in self.eng_stopwords]

        final_txt = ' '.join(txt)

        return final_txt
```

4. **Lemmatization:** Lemmatization is a very important and popular step carried out before any text analysis process. It can help us standardize the text columns and group together words that are similar to each other like `dance` and `dancing`. In the lemmatization process, the `ing` part of the word will be removed and both the words will now be considered the same in any future analysis that is carried out after lemmatization. This can help improve feature extraction from text, ensure better standardization of words during the vectorization process, and also improve the comprehension of text. It is similar to the `stemming` process but stemming does not ensure that the resultant words are meaningful and just utilizes rules to remove the suffix from words. While this can make stemming a faster process, `lemmatization can be a more thorough process and result in meaningful words, thus ensuring better comprehension`. Hence, in the final step of the text cleaning process, we have used lemmatization.  
The following code snippet was used for performing this analysis:
```python
    def _tokezine_lemmatize(txt):
        '''
        helper method for tokenizing and lemmatizing
        '''
        tokenized_txt = word_tokenize(txt)

        wordnet_lemmatizer = WordNetLemmatizer()

        lemmatized_txt = [wordnet_lemmatizer.lemmatize(w) for w in tokenized_txt]

        final_txt = ' '.join(lemmatized_txt)

        return final_txt
```

<br>

Examples of a few comments from before and after applying the above mentioned 4-step cleaning process has been shown below:

`Examples of Comments Before Cleaning:`
<pre>
- Oh, I never realized it was so easy, why had I, and every other lonely person on earth never thought of that before?

- And even wars such as Egyptian Unification war, a whopping 5316 years ago (Ish)

- I think he misspelled "Speed"
</pre>

`Examples of Comments after cleaning with the 4-Step cleaning process mentioned above:`
<pre>
- oh never realized easy every lonely person earth never thought

- even war egyptian unification war whopping year ago ish

- think misspelled speed
</pre>


<br>

**Cleaning Scraped HTML for User Data and Creating a DataFrame**

A cleaning process has also been carried out for the user data that was scraped form Reddit `(Source 2 Data mentioned above)`. This was carried out because the user information scrapred from Reddit was not always clean and not always in the correct format and hence some basic cleaning process were carried out. The following cleaning steps were carried out:  

1. **Cleaning Post/Comment Karma and Converting to Numeric Data Type**: The post and comment karma numbers were stored as text format and had comma separations within the numbers. We removed the commas from and converted the numbers from text format to numeric format. The missing values were then imputed with 0 values.  

Code snippet used for cleaning:
```python
if extracted_elem is not None:
    comment_karma = extracted_elem.text.replace(',', '')
    comment_karma = int(comment_karma)
else:
    comment_karma = 0
```

Examples:
```
Original Comment Karma:  <span class="karma comment-karma">355,040</span>

Cleaned Comment Karma:  355040
```

2. **Converting User Joining Date from Text to Date Type**: The user joining date was extracted and converted from a string format to a date time format by parsing the string as date time using the `dateutils` library. Additionally, all missing dates were imputed as `17th March 2023`. 

Code snippet used for cleaning:
```python
if join_date is not None:
    join_date = dp.parse(join_date['datetime']).date()
else:
    join_date = dp.parse('17 March 2023').date()
```

Examples:
```
Original Date String:  '2015-01-29T16:58:17+00:00'

Cleaned Date:  datetime.date(2015, 1, 29)
```

3. **Extracting User's List of Gilded Subreddits and Storing as a list**: All the list of user's subreddits where they either have a gilded post or gilded comment has been extracted and the extracted text has been stored as a list of strings. For cases where the user does not have any gilded posts or comments in any subreddit, it has been imputed with an empty list. 

Code snippet used for cleaning:
```python
all_gilded_posts = html_result.find_all(
    'div', attrs={'class':['gilded', 'thing']}
)
num_gilded_posts = len(all_gilded_posts)


### extract the list of subreddits the user received awards in
gilded_subreddits_list = [i['data-subreddit'] for i in all_gilded_posts]
```

Examples:
```
['ContagiousLaughter',
  'instant_regret',
  'pics',
  'gaming',
  'politics',
  'Showerthoughts']
```



<br>

**Data Cleaning after Vectorization**

We ended up creating two data sets, one for the comments section and the other for the parent_comments section. We used tf-idf vectorizer to convert the text data into different tokens that are going to aid us in fitting predictive models for the next part of the project. The same cleaning procedures were used for both the data set. The screenshot of the two sample data sets is shown below. The first screenshot of the sample comments and the second screenshot is for the parent comments. 
![](https://github.com/msis5223-pds2-2023spring/project-deliverable-1-cia/blob/main/assets/sample%20comments%20screenshot.png)
![](https://github.com/msis5223-pds2-2023spring/project-deliverable-1-cia/blob/main/assets/sample%20parent%20comments%20screenshot.png)
As seen in the two screenshots, there were quite a few null values in the data set. We realised that a few of these values occured due to the exporting of the data frames into a csv and then importing it back to perform further operations. We have dropped this null values but have kept a note of the index for which we are dropping these values as we need to consolidate the data later on. No treatment of the extreme values were so that no bias were introduced in the dataset. We are already vectorizing the text data, that itself is an operation that introduces some sort of biasedness. 


<br>

### Data Transformation
There we three kinds of data transformations carried out. These transformations have been explained below.   

**Text Data Transformation (TF-IDF Vectorization)**  

The most noticeable transformation out of all the transformations was converting the comments into tokens and then followed by creating TFIDF vectors for all the top occuring words. The vectors were limited to the top 70 words based on frequency and we limited to an upper limit n-gram of 3 and a lower limit n-gram of 1. This means that we will have individual words like `world`, `politics`, etc. being treated as individual vectors but we could also have words pairs and triplets like `yeah, sure` or `i am fine` being treated as individual word vectors. We wanted to be able to capture some semantic meaning and hence we decided to utilize this range of single words to word triplets. These steps were carried out after the text data cleaning steps mentioned above were performed.  

The reason for performing this vectorization steps was because most machine learning and statistical algorithms will not be able to parse text data directly and we will have to convert them into some form of numeric measure or numeric vectors on which we can apply machine learning algorithms to identify useful patterns. The vecotorization step has been used to perform this step and the numeric metric we have used for the vectors is `TFIDF score (Term Frequency - Inverse Document Frequency)`. There are different ways to create vectors with the most basic one being `Count Vectorization` where we simply calculate the count of occurences of each word. `TFIDF score is a slightly more robust and better metric` because this metric applies a weight to the occurence of each word based on how frequent or rare they are in the entire corpus. Hence, based on the TFIDF score, words that are very frequent in a document but very rare in the corpus will have higher weights. This can help us avoid giving too high weightage to very common and generic words like `book`, `car`, etc. by penalizing them and providing higher weightage to words that are generally rare in the corpus. `TFIDF Score` is usually calculated as a product of the term frequency (proportion of frequency of a word in a document) and the inverse document frequency (number of documents in a corpus divided by the total number of documents in the corpus where the term occurs. this value is usually log scaled.)

Hence, it can be represented as:

$\text{TF-IDF}(t, d, D) = \text{tf}(t, d) \times \text{idf}(t, D)$

where:

* $t$ is a term (word) in the document
* $d$ is a specific document in the corpus
* $D$ is the entire corpus
* $\text{tf}(t, d)$ is the term frequency of term $t$ in document $d$
* $\text{idf}(t, D) = \log{\frac{N}{n_t}}$ is the inverse document frequency of term $t$ across the entire corpus $D$, where:
    * $N$ is the total number of documents in the corpus
    * $n_t$ is the number of documents in the corpus that contain term $t$


Code Snippet Used for Vectorization:
```python
    tfidf = TfidfVectorizer(
        ngram_range=(1,3),
        max_features=70,
        strip_accents='unicode',
        use_idf=True,
        smooth_idf=True,
        min_df=0.0,
        max_df=0.80
    )

    comments_tfidf = tfidf.fit_transform(
        self.df['cleaned_'+self.text_column_name]
    )
```

The output from the above mentioned process generates a very high-dimensional and a sparse dataset. Hence, we will later be applying data reduction methods on it to reduce the dimensionality of the datasets.     
Sample Output from the Vectorization Process:    
![](assets/data_transform_1.png)

<br>

**Transformation for Numeric Metrics**

Some of the numeric metrics like `score`, `ups`, `comment_karma`, and `post_karma` were found to have high values of right-skewness. These variables were transformed to make the distribution more even and help us in the predictive modeling phase later down the road. The before and after transformation summary of a few metrics like `skewness`, `kurtosis`, and `standard deviation` along with the transformation applied have been shown in the tables below. We can see that before the transformation, the skewness and kurtosis values are very extreme. We can also see that after the `log transformations` the metrics have improved drastically the both the skewness and kurtosis are now quite close to zero.

<table>
    <tr>
        <td colspan="6">Before Transformation</td>
    </tr>
    <tr>
        <td>#</td>
        <td>Variable Name</td>
        <td>Transformation Applied</td>
        <td>Skewness</td>
        <td>Kurtosis</td>
        <td>Standard Dev</td>
    </tr>
    <tr>
        <td>1</td>
        <td>score</td>
        <td>None</td>
        <td>30.24991089</td>
        <td>1258.073068</td>
        <td>77.02079734</td>
    </tr>
    <tr>
        <td>2</td>
        <td>ups</td>
        <td>None</td>
        <td>32.69338315</td>
        <td>1456.851374</td>
        <td>68.00517955</td>
    </tr>
    <tr>
        <td>3</td>
        <td>comment_karma</td>
        <td>None</td>
        <td>7.827710191</td>
        <td>114.4187445</td>
        <td>104140.5426</td>
    </tr>
    <tr>
        <td>4</td>
        <td>post_karma</td>
        <td>None</td>
        <td>66.83917111</td>
        <td>6445.039043</td>
        <td>75630.51194</td>
    </tr>
</table>

<br>

<table>
    <tr>
        <td colspan="6">After Transformation</td>
    </tr>
    <tr>
        <td>#</td>
        <td>Variable Name</td>
        <td>Transformation Applied</td>
        <td>Skewness</td>
        <td>Kurtosis</td>
        <td>Standard Dev</td>
    </tr>
    <tr>
        <td>1</td>
        <td>score</td>
        <td>Log</td>
        <td>-0.973381061</td>
        <td>2.937969446</td>
        <td>1.844805272</td>
    </tr>
    <tr>
        <td>2</td>
        <td>ups</td>
        <td>Log</td>
        <td>-0.993396973</td>
        <td>2.793959394</td>
        <td>1.864830493</td>
    </tr>
    <tr>
        <td>3</td>
        <td>comment_karma</td>
        <td>Log</td>
        <td>-0.692805602</td>
        <td>-1.344020278</td>
        <td>6.812369526</td>
    </tr>
    <tr>
        <td>4</td>
        <td>post_karma</td>
        <td>Log</td>
        <td>-0.382880743</td>
        <td>-1.470978387</td>
        <td>5.720276132</td>
    </tr>
</table>

<br>

**Transformations of the Vectorized Values**  

We wanted to get the entirety of the numerical data in the same scale and therefore we wanted to normalize the data using the Z-score transformation. This ensures that the mean of the variables become 0 and the standard deviations are 1. This helps us in creating the covariance matrix and understand the linear spread of the data amongst variables. The following are the screenshots of the sample datasets of comments and parent comments respectively. 
![](https://github.com/msis5223-pds2-2023spring/project-deliverable-1-cia/blob/main/assets/normalized%20data%20comments.png)
![](https://github.com/msis5223-pds2-2023spring/project-deliverable-1-cia/blob/main/assets/normalized%20data%20parent%20comments.png)
Here, we can see that a lot of the entries have same values within a column for different columns. This is because a lot of the entries had the value zero before normalization and therefore to make sure that the overall column mean is 0 along with the fact that the column standard deviation should be 1 a lot of the entries have the same value. 


<br>

### Data Reduction
In the two datasets (comments and parent comments) that we just described, each of them had 73 columns and over 74000 rows. We needed to reduce the data in order to understand the data better and at the same time fitting algorithms on a smaller data is easier. And therefore we performed `Principal Component Analysis (PCA)` on each of the two datasets. Following is the detailed process of PCA performed on the two datasets:
1. Comments Data:
First, we dropped the columns that had the original and the cleaned text data in them along with the unique 'id' variable. After the aforementioned Z-score was done, we looked at the eigen values to understand the variance explained by each of the eigen vectors (and therefore each of the Principal Components). The cumulative sum of the eigen values gave us an idea about the percentage of variance explained based on the number of principal components. Below is the scree plot that showcases the optimum number of principal components to select based on the variance explained. 
![](https://github.com/msis5223-pds2-2023spring/project-deliverable-1-cia/blob/main/assets/Scree%20plot%20comments.png)
Based on this plot (and computational feasibility) we have chosen 20 principal components that explain 35.42% of the variance. After the PCA was done, we added back the columns with text along with the 'id' column to get a better understanding of the data. 

2. Parent Comments Data:
First, we dropped the columns that had the original and the cleaned text data in them along with the unique 'id' variable. After the aforementioned Z-score was done, we looked at the eigen values to understand the variance explained by each of the eigen vectors (and therefore each of the Principal Components). The cumulative sum of the eigen values gave us an idea about the percentage of variance explained based on the number of principal components. Below is the scree plot that showcases the optimum number of principal components to select based on the variance explained. 
![](https://github.com/msis5223-pds2-2023spring/project-deliverable-1-cia/blob/main/assets/scree%20plot%20parent%20comments.png)
Based on this plot (and computational feasibility) we have chosen 15 principal components that explain 26.33% of the variance. After the PCA was done, we added back the columns with text along with the 'id' column to get a better understanding of the data. 

### Data Consolidation

In this final step of our data prepration process, we will be combining all the data generated from different steps of our data cleaning and transformation pipeline along with the original base dataset and the scraped dataset to create a final master dataset that will be used for all further predictive analytics process. 

After performing PCA on two data sets (comments data and parent comments data) we have combined them to make one final dataset. We used the `id` as a primary key to merge these two data sets and an inner join was used to keep just the common records among the two datasets. The following code snippet shows how this process was done.

```python
df_combined = pd.merge(
    df_reduced_comment,
    df_reduced_p_comment,
    how='inner',
    on = 'id'
)
```

After the above process, we used the `id` variable to map the `label` to each row that signifies whether the comments and parent comments are sarcastic or not. This variable was obtained from the `base dataset`. Along with this, other numeric metrics obtained from the base dataset like `score`, `ups`, `downs`, `date`, etc. were also joined. This join was also an inner join.

```python
df_final = pd.merge(
    df_combined,
    df_base_1,
    on='id',
    how='inner'
)
```

Finally, in the final step of the process, we joined the `users dataset` with the merged datasets obtained from the above merging process. This allowed us to obtain user level metrics like `comment_karma`, `post_karma`, `gilded_subreddits`, `join_date`, etc. For this join, we used the key `author` which is the username and this join was also an inner join.

```python
df_final_v2 = pd.merge(
    df_final,
    user_data, 
    on = 'author', 
    how = 'inner'
)
```


Below is the screenshot of a sample of the final data set. 

![](assets/final_data_screenshot_v2.png)

<br>

![](assets/final_data_screenshot_v3.png)

<br>

### Data Dictionary

The following table shows the data dictionary for the final master table generated by consolidating the data obtained from various different sources - Kaggle Dataset, Scraped Data from Reddit and the data generated from the different data cleaning processes that were carried out.

<table>
    <thead>
        <tr>
            <th>Attribute Name</th>
            <th>Description</th>
            <th>Data Type</th>
            <th>Source</th>
            <th>Data</th>
            <th>Example</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>id</td>
            <td>Unique Identifier for each comment</td>
            <td>integer</td>
            <td>Internally generated</td>
            <td><a href='data/final_data/final_master_data.csv'>data/final_data/final_master_data.csv</a></td>
            <td>10, 89, 1785, 14330</td>
        </tr>
        <tr>
            <td>cleaned_comment</td>
            <td>Cleaned version of the comments (lowercase conversion, lemmatization, special characters removed)</td>
            <td>string</td>
            <td>Internally generated</td>
            <td><a href='data/final_data/final_master_data.csv'>data/final_data/final_master_data.csv</a></td>
            <td>two thing mutually exclusive lol; first trillionaire time trump office;people voted trump</td>
        </tr>
        <tr>
            <td>comment</td>
            <td>Comment made by the author or user</td>
            <td>string</td>
            <td><a href='https://www.kaggle.com/datasets/danofer/sarcasm'>https://www.kaggle.com/datasets/danofer/sarcasm</a></td>
            <td><a href='data/final_data/final_master_data.csv'>data/final_data/final_master_data.csv</a></td>
            <td>Are those two things mutually exclusive... lol?; Will be first trillionaire by the time Trump is out of office;</td>
        </tr>
        <tr>
            <td>comp_00_comments, comp_01_comments, …, comp_19_comments"</td>
            <td>20 Components obtained from the PCA Data Reduction Process applied on the TFIDF Vectorized Comments Data</td>
            <td>float</td>
            <td>Internally generated</td>
            <td><a href='data/final_data/final_master_data.csv'>data/final_data/final_master_data.csv</a></td>
            <td>0.0006219, -0.000522, -0.00065415</td>
        </tr>
        <tr>
            <td>cleaned_parent_comment</td>
            <td>Cleaned version of the parent comments where the comments were made (lowercase conversion, lemmatization, special characters removed)</td>
            <td>string</td>
            <td>Internally generated</td>
            <td><a href='data/final_data/final_master_data.csv'>data/final_data/final_master_data.csv</a></td>
            <td>he 's get richer; executive advisor trump refuting climate change claim earth year old;</td>
        </tr>
        <tr>
            <td>parent_comment</td>
            <td>Parent Comment for the comment thread where the comment was made</td>
            <td>string</td>
            <td><a href='https://www.kaggle.com/datasets/danofer/sarcasm'>https://www.kaggle.com/datasets/danofer/sarcasm</a></td>
            <td><a href='data/final_data/final_master_data.csv'>data/final_data/final_master_data.csv</a></td>
            <td>he's about to get richer.; An executive advisor to Trump,while refuting climate change, claims the earth is only 5,500 years old.</td>
        </tr>
        <tr>
            <td>comp_00_p_comments, comp_01_p_comments, …, comp_14_p_comments</td>
            <td>15 Components obtained from the PCA Data Reduction Process applied on the TFIDF Vectorized Comments Data</td>
            <td>float</td>
            <td>Internally generated</td>
            <td><a href='data/final_data/final_master_data.csv'>data/final_data/final_master_data.csv</a></td>
            <td>0.001228, -0.002156, -0.013206</td>
        </tr>
        <tr>
            <td>label</td>
            <td>Target Label (1/0) indicating Sarcastic Comment (1) or Non Sarcastic Comment (0) </td>
            <td>integer</td>
            <td><a href='https://www.kaggle.com/datasets/danofer/sarcasm'>https://www.kaggle.com/datasets/danofer/sarcasm</a></td>
            <td><a href='data/final_data/final_master_data.csv'>data/final_data/final_master_data.csv</a></td>
            <td>Binary values 1 or 0</td>
        </tr>
        <tr>
            <td>author</td>
            <td>Reddit username of the redditor that made the comment</td>
            <td>string</td>
            <td><a href='https://www.kaggle.com/datasets/danofer/sarcasm'>https://www.kaggle.com/datasets/danofer/sarcasm</a></td>
            <td><a href='data/final_data/final_master_data.csv'>data/final_data/final_master_data.csv</a></td>
            <td>pb2crazy, ThisIsNotKimJongUn</td>
        </tr>
        <tr>
            <td>subreddit</td>
            <td>The name of the Subreddit on Reddit where the comment was made</td>
            <td>string</td>
            <td><a href='https://www.kaggle.com/datasets/danofer/sarcasm'>https://www.kaggle.com/datasets/danofer/sarcasm</a></td>
            <td><a href='data/final_data/final_master_data.csv'>data/final_data/final_master_data.csv</a></td>
            <td>worldnews, AskReddit, politics</td>
        </tr>
        <tr>
            <td>score</td>
            <td>Score is a metric that is derived from downvotes and upvotes. It is simply the difference between the number of upvotes and downvotes.</td>
            <td>integer</td>
            <td><a href='https://www.kaggle.com/datasets/danofer/sarcasm'>https://www.kaggle.com/datasets/danofer/sarcasm</a></td>
            <td><a href='data/final_data/final_master_data.csv'>data/final_data/final_master_data.csv</a></td>
            <td>3, 2, 0</td>
        </tr>
        <tr>
            <td>ups</td>
            <td>Number of upvotes received on the comment (similar to likes on other social media platforms)</td>
            <td>integer</td>
            <td><a href='https://www.kaggle.com/datasets/danofer/sarcasm'>https://www.kaggle.com/datasets/danofer/sarcasm</a></td>
            <td><a href='data/final_data/final_master_data.csv'>data/final_data/final_master_data.csv</a></td>
            <td>2, -1, 3</td>
        </tr>
        <tr>
            <td>downs</td>
            <td>
            Number of downvotes received on the comment (similar to dislikes on other social media platforms)
            https://reddit.zendesk.com/hc/en-us/articles/7419626610708-How-does-voting-work-on-Reddit-
            </td>
            <td>integer</td>
            <td><a href='https://www.kaggle.com/datasets/danofer/sarcasm'>https://www.kaggle.com/datasets/danofer/sarcasm</a></td>
            <td><a href='data/final_data/final_master_data.csv'>data/final_data/final_master_data.csv</a></td>
            <td>2, -1, 3</td>
        </tr>
        <tr>
            <td>date</td>
            <td>The date on which the comment was posted</td>
            <td>date</td>
            <td><a href='https://www.kaggle.com/datasets/danofer/sarcasm'>https://www.kaggle.com/datasets/danofer/sarcasm</a></td>
            <td><a href='data/final_data/final_master_data.csv'>data/final_data/final_master_data.csv</a></td>
            <td>2016-11-01; 2016-12-03; 2016-12-02;</td>
        </tr>
        <tr>
            <td>post_karma</td>
            <td>
                Scraped Data. Historical Karma accumulated by the user on Reddit through posts they have created on Reddit 
                (Note: Karma is the points system on Reddit. Reddit calculates it as a factor of upvotes and downvotes)
                https://reddit.zendesk.com/hc/en-us/articles/204511829-What-is-karma-
            </td>
            <td>integer</td>
            <td><a href='https://old.reddit.com/'>https://old.reddit.com/</a></td>
            <td><a href='data/final_data/final_master_data.csv'>data/final_data/final_master_data.csv</a></td>
            <td>0, 10, 5000</td>
        </tr>
        <tr>
            <td>comment_karma</td>
            <td>
                Scraped Data. Historical Karma  accumulated by the user on Reddit through comments they have made on Reddit.
                (Note: Karma is the points system on Reddit. Reddit calculates it as a factor of upvotes and downvotes)
            </td>
            <td>integer</td>
            <td><a href='https://old.reddit.com/'>https://old.reddit.com/</a></td>
            <td><a href='data/final_data/final_master_data.csv'>data/final_data/final_master_data.csv</a></td>
            <td>20, 56000, 0</td>
        </tr>
        <tr>
            <td>join_date</td>
            <td>Scraped Data. The date on which the user joined reddit</td>
            <td>date</td>
            <td><a href='https://old.reddit.com/'>https://old.reddit.com/</a></td>
            <td><a href='data/final_data/final_master_data.csv'>data/final_data/final_master_data.csv</a></td>
            <td>2011-11-01; 2012-12-03; 2008-12-02;</td>
        </tr>
        <tr>
            <td>gilded_posts</td>
            <td>
                Scraped Data. The number of gilded posts or comments the user has on reddit. 
                (Note: A post or comment is considered gilded if they receive an award for their post or comment from other users.)
                https://reddit.zendesk.com/hc/en-us/articles/360043034132-What-are-awards-and-how-do-I-give-them-
            </td>
            <td>integer</td>
            <td><a href='https://old.reddit.com/'>https://old.reddit.com/</a></td>
            <td><a href='data/final_data/final_master_data.csv'>data/final_data/final_master_data.csv</a></td>
            <td>10, 5, 0</td>
        </tr>
        <tr>
            <td>gilded_post_subreddits</td>
            <td>
                Scraped Data. The number of gilded posts or comments the user has on reddit. 
                A post or comment is considered gilded if they receive an award for their post or comment from other users.
            </td>
            <td>list</td>
            <td><a href='https://old.reddit.com/'>https://old.reddit.com/</a></td>
            <td><a href='data/final_data/final_master_data.csv'>data/final_data/final_master_data.csv</a></td>
            <td>['AskReddit', 'ContagiousLaughter', 'instant_regret']</td>
        </tr>
        <tr>
            <td>top_gilded_subreddit</td>
            <td>The name of the subreddit where the user has the highest number of gilded posts/comments</td>
            <td>string</td>
            <td>Internally generated</td>
            <td><a href='data/final_data/final_master_data.csv'>data/final_data/final_master_data.csv</a></td>
            <td>ContagiousLaughter, AskReddit</td>
        </tr>
        <tr>
            <td>gilded_unique_subs_count</td>
            <td>Total number of unique subreddits where the user has any gilded post or comment</td>
            <td>integer</td>
            <td>Internally generated</td>
            <td><a href='data/final_data/final_master_data.csv'>data/final_data/final_master_data.csv</a></td>
            <td>0, 5, 7</td>
        </tr>
        <tr>
            <td>log_score</td>
            <td>Log Transformed version of the above mentioned score variable</td>
            <td>float</td>
            <td>Internally generated</td>
            <td><a href='data/final_data/final_master_data.csv'>data/final_data/final_master_data.csv</a></td>
            <td>0.6936, 1.0989, -6.907</td>
        </tr>
        <tr>
            <td>log_ups</td>
            <td>Log Transformed version of the above mentioned ups variable</td>
            <td>float</td>
            <td>Internally generated</td>
            <td><a href='data/final_data/final_master_data.csv'>data/final_data/final_master_data.csv</a></td>
            <td>1.7932, 3.09, -1.372</td>
        </tr>
        <tr>
            <td>log_comment_karma</td>
            <td>Log Transformed version of the above mentioned comment_karma variable</td>
            <td>float</td>
            <td>Internally generated</td>
            <td><a href='data/final_data/final_master_data.csv'>data/final_data/final_master_data.csv</a></td>
            <td>0.343, 0.03672, -1.289</td>
        </tr>
        <tr>
            <td>log_post_karma</td>
            <td>Log Transformed version of the above mentioned post_karma variable</td>
            <td>float</td>
            <td>Internally generated</td>
            <td><a href='data/final_data/final_master_data.csv'>data/final_data/final_master_data.csv</a></td>
            <td>9.832, 1.8906, -1.005</td>
        </tr>
    </tbody>
</table>

<br>
<br>

## Descriptive Statistics and Analysis

### Summary Statistics

|          |   mean    | median |     standard deviation     |  maximum  | mininimum |
|:--------:|:---------:|:------:|:----------:|:-----:|:---:|
|  label   |  0.475338 |   0    | 0.4993948  |   1   |  0  |
|  score   | 8.6171378 |   1    | 77.0172307 |  4981 | -93 |
|   ups    | 6.7029872 |   1    | 68.0020219 |  4776 | -93 |
|  downs   | -0.163950 |   0    | 0.3702328  |   0   | -1  |
|post_karma|7461.017403|  250   | 7.563051e+04|9252884|  0  |
|comment_karma|42208.897815| 7416 |1.041405e+05|2963717|-100|
|gilded_posts|  1.605206 |   0    |3.692283e+00|  25   |  0  |
|gilded_unique_subs_count| 1.753667| 1| 2.002959e+00| 23| 1|


### Univariate Analysis

**label** *(Target Variable - Binary)*<br>

<img src = 'assets/visualization/univariate_label.png' width=700, height=400><br>

The target variable, binary indicator of sarcasm, is somewhat balanced.<br>

**score**<br>

<img src = 'assets/visualization/univariate_score.png' width=700, height=400><br>

`score` is heavily right-skewed, transformation such as log transformation might be needed to convert into a distribution closer to normal to avoid bias.<br>

**ups**<br>

<img src = 'assets/visualization/univariate_ups.png' width=700, height=400><br>

`ups` is heavily right-skewed, transformation such as log transformation might be needed to convert into a distribution closer to normal to avoid bias.<br>

**downs**<br>

<img src = 'assets/visualization/univariate_downs.png' width=700, height=400><br>

There are more comments with downvotes than those that are not.<br>

**post_karma**<br>

<img src = 'assets/visualization/univariate_post_karma.png' width=700, height=400><br>

`post_karma` is heavily right-skewed, transformation such as log transformation might be needed to convert into a distribution closer to normal to avoid bias.

**comment_karma**<br>

<img src = 'assets/visualization/univariate_comment_karma.png' width=700, height=400><br>

`comment_karma` is heavily right-skewed, transformation such as log transformation might be needed to convert into a distribution closer to normal to avoid bias.

**gilded_posts**<br>

<img src = 'assets/visualization/univariate_gilded_posts.png' width=700, height=400><br>

`gilded_posts` is heavily right-skewed, transformation such as log transformation might be needed to convert into a distribution closer to normal to avoid bias.

**gilded_unique_subs_count**<br>

<img src = 'assets/visualization/univariate_gilded_unique_subs_count.png' width=700, height=400><br>

`gilded_unique_subs_count` is heavily right-skewed, transformation such as log transformation might be needed to convert into a distribution closer to normal to avoid bias.

**Proportion of Words in Parent Sarcastic Comments**

<img src = "assets/Proportion%20of%20words%20in%20Sarcastic%20parent%20comments.png"  width=700, height=400><br>

A proportion graph depicting the frequency of certain words in a corpus can reveal which words are most common or over-represented in the dataset. In sarcastic parent comments, the words 'people' and 'Trump' are the most prevalent, with 'time' and 'world' ranking third and fourth in frequency. It appears that the words 'people' and 'Trump' may be over-represented, as most of the sarcastic sentences originate from the politics subreddit. A pattern is also observed wherein political and offensive words have a significant proportion in sarcastic statements. However, the data still has a usable distribution since most parent comments contain these words and do not appear to be over-represented in the data, making it suitable for further modeling.<br>

**Proportion of Words in Sarcastic Comments**

<img src = "assets/Proportion%20of%20words%20in%20Sarcastic%20Comments.png"  width=700, height=400><br>

Looking at the proportion of words in sarcastic comments, it is observed that the words 'people' and 'Trump' are the most prevalent, similar to sarcastic parent comments. The majority of comments that contain these words are found to be sarcastic and are typically found in the 'politics' subreddit. Other words also appear in the distribution, but at lower proportions. Some of these words clearly demonstrate that the sentence could be sarcastic; however, the proportion of such words is low in the overall sentences.<br>

A pattern was also observed in which political and offensive words had a significant proportion in sarcastic statements, which is the same pattern observed in the parent comments. Most sarcastic sentences tend to include these types of words, providing evidence that the dataset used does not have over-represented words and is suitable for further modeling.<br>


### Multivariate Analysis

**Sarcasm Rate and Count on Reddit from 2009 to 2016**

<img src = 'assets/visualization/multivariate_01.png' width=700, height=400><br>

The plot displays the trend of sarcasm on Reddit from 2009 to 2016, with sarcasm rate and count being the two main variables of interest. The data reveals that the sarcasm rate, which represents the proportion of sarcastic comments or posts compared to the total number of comments or posts, has been decreasing slightly over time, dropping from above 0.5 to less than 0.5 at the end of 2016. This suggests that users may have become more cautious about using sarcasm on the platform or that the moderation policy has been stricter in identifying and removing sarcastic content.<br>

However, the plot also shows that the count of sarcasm has increased exponentially over time, indicating that the volume of sarcastic comments and posts has increased despite the decrease in the sarcasm rate. This suggests that while the proportion of sarcastic content may have decreased, the overall volume of content on the platform has increased significantly. The plot highlights the importance of considering both the sarcasm rate and count when analyzing the trend of sarcasm on Reddit, as changes in one variable may not necessarily reflect the trend of the other variable.<br>

**Sarcasm Rate and Count on Reddit by User Tenure**

<img src = 'assets/visualization/multivariate_02.png' width=700, height=400><br>

The plot of sarcasm rate and count on Reddit by user tenure shows interesting trends that shed light on the dynamics of sarcasm among users. One observation is that Sarcasm Rate fluctuated as user tenure increased beyond 6 years, indicating that highly tenured users may impact sarcasm rate greatly due to the lack of highly tenured users in the dataset. <br>

However, despite these fluctuations, the overall Sarcasm Rate line hovers close to 0.5, indicating that sarcasm among users maintains regardless of user tenures. This is an interesting finding as it suggests that sarcasm is a common form of communication on Reddit, regardless of whether users are new to the platform or not. It also implies that users may not necessarily become less sarcastic as they spend more time on the platform.<br>

In summary, the plot of Sarcasm Rate and Count on Reddit by User Tenure highlights the complexity of sarcasm as a communication style on the platform. While highly tenured users may have a greater impact on sarcasm rates, the overall sarcasm rate remains relatively stable over time, indicating that it is a prevalent form of communication among Reddit users.<br>

**Association between Sarcasm Rate and Subreddit**

<img src = 'assets/visualization/multivariate_03.png' width=700, height=400><br>

The association between sarcasm rate and subreddit can reveal interesting patterns in the way users interact in different forums. From the plot, we can observe that the subreddits politics and worldnews have an average sarcasm rate of around 0.6, while askreddit has a sarcasm rate of around 0.35. This suggests that political and news discussions tend to attract more sarcastic comments, perhaps due to the contentious and polarizing nature of the topics.<br>

Additionally, we can see that askreddit has the highest average Reddit score (based on the size of the points) of around 10, followed by worldnews at 6.6 and politics at 6.13. Interestingly, we can also observe that higher scores tend to correspond with lower sarcasm rates. This could indicate that users in these subreddits are more likely to engage in genuine discussions and share valuable insights, rather than resorting to sarcasm as a means of communication.<br>

Overall, this plot highlights the differences in the tone and culture of different subreddits, and how they may impact user behavior and interactions. It also provides insight into how the type of content and the community of a subreddit may influence the prevalence of sarcasm and other forms of communication.<br>

**Association between Sarcasm Rate and User's Top Gilded Subreddit**

<img src = 'assets/visualization/multivariate_05.png' width=700, height=400><br>

The plot of association between sarcasm rate and user's top gilded subreddit provides insights into the relationship between the content of subreddits and sarcasm. The data shows that neutral subreddits like AskReddit, AdviceAnimals, funny, and pics have a lower sarcasm rate, while politically-focused subreddits like politics and worldnews tend to have a higher sarcasm rate. Similarly, subreddits like todayilearned have a higher sarcasm rate despite being less politically focused. This could be due to the nature of the subreddit, which focuses on interesting and little-known facts. Users may feel more comfortable expressing sarcasm in this type of environment where the focus is not on serious political or social issues. Additionally, the sarcasm could be a way for users to express their skepticism or disbelief in some of the facts presented.<br>

Interestingly, the analysis also shows that the average gilded unique subreddit count and average gilded post count do not appear to have any significant association with sarcasm rate. This suggests that the number of gilded posts or the diversity of gilded subreddits that a user engages with does not necessarily impact their use of sarcasm.<br>

Overall, this analysis sheds light on the relationship between the content of subreddits and the use of sarcasm. The findings suggest that politically-focused subreddits tend to have a higher sarcasm rate, while neutral subreddits are less sarcastic. Additionally, the data shows that the number of gilded posts or subreddits that a user engages with does not appear to be a significant factor in their use of sarcasm.<br>

**Association between Sarcasm Rate and User's Top Gilded Subreddit**

<img src = 'assets/visualization/multivariate_04.png' width=700, height=400><br>

Empirically, there seems to be no clear association between sarcasm rate and user's post/comment karma based on the scatterplot. The majority of users have higher post karma counts than comment karma counts, indicating that they are more active in submitting posts rather than commenting on them. However, the distribution of sarcasm rates appears to be relatively uniform across the different levels of post/comment karma.

The regression lines on the scatterplot indicate that there is no significant relationship between sarcasm rate and post/comment karma. This means that users with high post or comment karma counts are not more likely to be sarcastic than those with lower karma counts. Therefore, it can be inferred that users' engagement on Reddit, as measured by their karma counts, does not necessarily affect their tendency to use sarcasm.<br>

Overall, the scatterplot suggests that sarcasm is a prevalent form of communication among Reddit users, regardless of their level of engagement on the platform.<br>

## Text Mining and Sentiment Analysis

### Sentiment Analysis

`nrclex` library in Python was applied for sentiment analysis. 2 pairs of sentiments/emotions were picked to conduct sentiment analysis as follows:

1. `Trust` vs. `Fear`
2. `Disgust` vs. `Surprise`

It is worth pointing out that **affect frequency** of sentiments in a Reddit comment was applied in our analysis for comparison as well as feeding into future models as predictors. Affect frequency refers to the frequency with which emotional words or expressions are used in a given text, while normal frequency refers to the frequency of all words in the text. Affect frequency can offer several advantages over normal frequency in sentiment analysis and other natural language processing tasks. Firstly, it can help identify the emotional content of a text more accurately, as emotional words and expressions are more indicative of the sentiment and tone of the text than other words. This can result in more precise sentiment scores and a better understanding of the overall sentiment of the text. Secondly, affect frequency can help normalize language use and overcome imbalances in the occurrence of different types of words in a text. By using affect frequency, sentiment analysis models can provide more reliable and accurate results that are less affected by such imbalances. Finally, affect frequency can reveal trends and patterns in emotional language use over time or across different contexts. By analyzing the affect frequency of a text over a period of time or in different domains, sentiment analysis models can provide valuable insights into how emotions are expressed and perceived in different contexts, and how they change over time. Overall, using affect frequency can enhance the accuracy and reliability of sentiment analysis, and provide deeper insights into the emotional content of a text.

1. **`Trust` vs. `Fear`**<br>

There are several reasons why `trust` and `fear` may be a good choice of emotions to use for sentiment analysis in a Reddit sarcasm detection project. First, `trust` and `fear` are widely recognized as fundamental emotions that are commonly expressed in human communication. They are also emotions that are often associated with sarcasm, irony, and other forms of subtle and nuanced language use, which makes them particularly relevant for sarcasm detection. Second, `trust` and `fear` are often seen as opposing emotions, which means that their presence in a text can provide a valuable contrast that can help identify the underlying sentiment and tone of the message. Finally, `trust` and `fear` are emotions that are likely to be expressed in a variety of contexts, which makes them suitable for use in a broad range of Reddit conversations and topics. Overall, selecting `trust` and `fear` as emotions for sentiment analysis in a Reddit sarcasm detection project can provide a useful and reliable way to identify the underlying sentiment and tone of text, and help detect sarcastic or ironic expressions.

The frequency bar chart and wordcloud top 20 most common `trust` words in our dataset is as shown below:<br>

<img src = 'assets/topic_sentiment_analysis/01_trustfear.png' width=700><br>

<img src = 'assets/topic_sentiment_analysis/02_trustfear.png' width=400><br>

Good, money, and white appeared to be the three most common words for `trust`.<br>

On the other hand, the frequency bar chart and wordcloud top 20 most common `fear` words in our dataset is as shown below:<br>

<img src = 'assets/topic_sentiment_analysis/03_trustfear.png' width=700><br>

<img src = 'assets/topic_sentiment_analysis/04_trustfear.png' width=400><br>

Bad, government, and war appeared to be the three most common words for `trust`.<br>

Comparison of average affect frequency between `fear` and `trust` in a Reddit comment with and without taking consideration of sarcasm label:<br>

<img src = 'assets/topic_sentiment_analysis/05_trustfear.png' width=700><br>

<img src = 'assets/topic_sentiment_analysis/06_trustfear.png' width=700><br>

By comparing the average affect frequency between `fear` and `trust` in a Reddit comment in general, `trust` was more common at average affect frequency of 0.086 per comment as compared to `fear` at 0.0526 per comment. The higher average affect frequency for `trust` as compared to `fear` suggests that `trust` was a more commonly expressed emotion in Reddit comments than `fear`. However, when taking sarcasm label into account, both `fear` and `trust` were slightly more common when there was no sarcasm detected in the comments. This finding may suggest that in sarcastic comments, emotions other than fear and trust are slightly more commonly used to express sarcasm. Alternatively, it could indicate that in non-sarcastic comments, fear and trust are more frequently expressed emotions as they convey sincerity and authenticity, whereas in sarcastic comments, people may use other emotions to convey a sense of irony or humor. This highlights the importance of considering contextual factors, such as sarcasm, when analyzing emotional language use in natural language processing tasks.

**Example Reddit Comments Containing `Fear` and `Trust`**

*Where sarcasm was present*

<img src = 'assets/topic_sentiment_analysis/13_validation.png' width=700><br>

*Where sarcasm was not present*

<img src = 'assets/topic_sentiment_analysis/14_validation.png' width=700><br>

The sentiments presented in the examples are closely related to the intention of the comment author. For instances, `fear` emotion can be found in comment related to political ideology and `trust` emotion can be found in comment related to employment or office relationship.  <br>


2. **`Surprise` vs. `Disgust`**<br>

Using the pair of emotions `surprise` vs. `disgust` in sentiment analysis for a Reddit sarcasm detection project can provide valuable insights into the emotional content of the comments. `Surprise` is often associated with positive emotions, such as excitement, delight, and amazement, whereas `disgust` is associated with negative emotions, such as aversion, revulsion, and contempt. By comparing the affect frequency of these two emotions in the comments, the sentiment analysis model can identify the overall emotional tone and sentiment of the comments, and potentially detect instances of sarcasm or irony that may be expressed through the use of unexpected or contradictory emotions. Additionally, `surprise` and `disgust` are often expressed through distinctive lexical and syntactic patterns, which can aid in the development of more accurate and effective sentiment analysis models. Therefore, the use of `surprise` vs. `disgust` in sentiment analysis can be a useful approach in detecting sarcasm and other nuanced forms of emotional expression in Reddit comments.

The frequency bar chart and wordcloud top 20 most common `surprise` words in our dataset is as shown below:<br>

<img src = 'assets/topic_sentiment_analysis/07_surprisedisgust.png' width=700><br>

<img src = 'assets/topic_sentiment_analysis/08_surprisedisgust.png' width=400><br>

Good, trump, and money appeared to be the three most common words for `surprise`.<br>

On the other hand, the frequency bar chart and wordcloud top 20 most common `disgust` words in our dataset is as shown below:<br>

<img src = 'assets/topic_sentiment_analysis/09_surprisedisgust.png' width=700><br>

<img src = 'assets/topic_sentiment_analysis/10_surprisedisgust.png' width=400><br>

Bad, government, and war appeared to be the three most common words for `disgust`.<br>

Comparison of average affect frequency between `surprise` and `disgust` in a Reddit comment with and without taking consideration of sarcasm label:<br>

<img src = 'assets/topic_sentiment_analysis/11_surprisedisgust.png' width=700><br>

<img src = 'assets/topic_sentiment_analysis/12_surprisedisgust.png' width=700><br>

By comparing the average affect frequency between `surprise` and `disgust` in a Reddit comment in general, both sentiments were having a similar average sentiment weight in a Reddit comment at about 0.03 which suggest that both emotions were as commonly being expressed as each other in our dataset. However, when taking sarcasm label into account, the results showed that the `surprise` emotion had a stronger association with non-sarcastic comments, while `disgust` was more associated with sarcasm when comparing the two emotions. This finding suggests that the use of surprise in a comment may be indicative of a sincere expression of emotion, whereas the use of disgust may be more likely to signal sarcasm or irony. <br>

**Example Reddit Comments Containing `Surprise` and `Disgust`**

*Where sarcasm was present*

<img src = 'assets/topic_sentiment_analysis/15_validation.png' width=700><br>

*Where sarcasm was not present*

<img src = 'assets/topic_sentiment_analysis/16_validation.png' width=700><br>

The sentiments presented in the examples are closely related to the intention of the comment author. For instances, `surprise` emotion can be found in comment praising a movie and `disgust` emotion can be found in comment related to feaces.  <br>

### Topic Analysis

Topic analysis is a useful approach for detecting sarcasm. By identifying the topics and themes present in a comment, topic analysis can provide contextual clues that can help distinguish between sarcastic and sincere statements. Furthermore, the combination of topic analysis with other techniques, such as sentiment analysis and affect frequency analysis done previously, can further improve the accuracy and effectiveness of sarcasm detection models.<br>

To conduct a topic analysis, a `CountVectorizer` was initialized with `max_df` of 0.8 indicating the maximum occurence of a term was 80% of the documents and `min_df` of 10 indicating the minimum number of documents a word must appear in was 10 documents. Then, a document-term matrix was created by fitting and transforming the text data using the vectorizer. The document-matrix was subsequently fitted into `Latent Dirichlet Allocation (LDA)` to create 4 topics. The resulted top 10 words for each topic are as shown:

<img src = 'assets/topic_sentiment_analysis/17_topic.png' width=700><br>

Based on the result above, the 4 topics from Reddit comments in our dataset can be summarized as follows:

- **Topic #0: Casual Conversations**

This topic seems to be centered around casual conversations on Reddit with terms like "guy", "make", "good", and "yeah" appearing frequently. The word "fuck" is likely used as a form of emphasis in casual speech. The word "white" could potentially refer to race or just be a common adjective used in conversation. The word "ca" is likely an abbreviation for "can't" or "cannot". Overall, this topic appears to be centered around everyday conversations that people might have.

- **Topic #1: Relationships and Personal Reflections**

This topic appears to focus on personal reflections and relationships. Words like "child," "love," and "woman" suggest discussions around family and romantic relationships. The words "reddit," "think," and "got" may indicate that these are personal stories being shared in an online forum.

- **Topic #2: Mixed Topics**

This topic could be related to government, particularly in the context of how government operates and how it can be improved. The word "best" may suggest a discussion of best practices, while the words "really" and "better" may indicate a desire for improvement. However, the other words in the topic, such as "nan" and "jerry," don't seem to fit with a political discussion. It's possible that this topic contains a mixture of discussions, or that the word "government" appeared incidentally in a few posts that were primarily about other discussions. 

- **Topic #3: Politics and Current Events**

This topic appears to be focused on politics and current events. Words like "government," "trump," and "people" suggest a discussion about policies and leaders. The words "great" and "sure" may indicate a positive or optimistic perspective on certain political events or outcomes. However, these words may bring sarcastic sentiments too. The word "forgot" may suggest that the conversation is centered around a past event or issue that has been overlooked.<br>

**Relationship between Topics and Respective Sentiments as well as Sarcastic Features**

| Topic | Fear | Trust | Disgust | Surprise | Label |
| -- | :-- | :-- | :-- | :-- | :-- |
| 0 | 0.053 | 0.078 | 0.032 | 0.027 | 0.53 |
| 1 | 0.047 | 0.087 | 0.030 | 0.025 | 0.37 |
| 2 | 0.058 | 0.092 | 0.032 | 0.022 | 0.40 |
| 3 | 0.053 | 0.089 | 0.029 | 0.048 | 0.56 |

- `Label` - Probability of sarcastic comments

Topic 3 that is related to politics and current events has the highest weight of sarcasm which aligns to our previous descriptive analytics where `politcs` and `worldnews` are the most sarcastic subreddits. Topic 0 has the second highest sarcasm weight as it is related to casual conversation and it makes sense as sarcasm could take place in normal day-to-day conversation according to human nature. On the contrary, Topic 1 is the least sarcastic. This makes sense too because typically people will be more sensitive when it comes to relationships and personal reflection because mental health and kindness are taken into consideration before leaving comment in related Reddit Post. Topic 2 is relatively low sarcastic but since it is related to discussion of mixed topic, no hard conclusion can be made.

- Average Affect Frequency of `Fear` and `Trust`

Topic 0 has the least amount of trust (0.078) and moderate fear (0.053) among all the topics. This suggests that the topic is centered around casual conversations and does not contain any particularly polarizing or controversial topics. Topic 1 has a moderate level of trust (0.087) and fear (0.047). This may suggest that the topic includes personal stories and reflections, which may evoke emotions but are not necessarily controversial. Topic 2 has the highest level of trust (0.092) and a slightly higher level of fear (0.058). This could be due to the potential political nature of the topic, which could be a source of anxiety or uncertainty for some people. Topic 3 has a moderate level of trust (0.089) and a higher level of fear (0.053), which suggests that the topic could be focused on more polarizing issues such as politics and current events.

- Average Affect Frequency of `Disgust` and `Surprise`

In terms of disgust, none of the topics have particularly high or low values, with the highest being 0.032 for topics 0 and 2. For surprise, the values are also relatively low for all topics, with the highest being 0.048 for topic 3. This suggests that the topics are not particularly surprising or unexpected in nature except for topic 3 as it is related to politics and current events that might be shocking.

In summary, the topic analysis is somewhat related to the sentiment analysis done previously. For example, when politcal views are brought up, `fear`, `trust`, and `surprise` emotions are relatively significant. When relationship and personal reflection elements are found in comments, only `trust` element appears to be significant. On the other hand, when it is just a conversation, `trust` carries the lowest weight while `disgust` appears to be slightly more significant as the language used might contain disgusting elements (Redditors are anonymous).<br>

### Named-Entity Recognition Analysis

- **Person Entity**

<img src = 'assets/topic_sentiment_analysis/19_NER_PERSON.png' width=700><br>

Politicians like "Trump", "Obama", and "Hilary" are mostly brought up in Reddit comments based on the graph above. Since most of the person entities in our dataset are political related, the sarcasm rate is high overall. This further supports the nature of Reddit where users are anonymous and they will be having a higher tendency to leave sarcastic comments discussing politicians.

- **Location Entity**

<img src = 'assets/topic_sentiment_analysis/20_NER_LOCATION.png' width=700><br>

The common locations being mentioned in the comments in our dataset are mostly countries or territories that are geo-politically controversial like "Israel", "Russia", and "China". The geo-political controversies may arise tons of sarcastic comments making the sarcasm rate high as shown in the graph. "America", on the other hand, acts as a global leader and Redditors seem to be very sarcastic to any event happening in the mentioned country.

- **Organization Entity**

<img src = 'assets/topic_sentiment_analysis/21_NER_ORGANIZATION.png' width=700><br>

Similarly to locations, the common organizations mentioned in the Reddit comments are highly related to politics and world events like "US", "ISIS", and "Muslim". It aligns to every analysis done previously where `politics` and `worldnews` subreddits are not only common in Reddits, but also having the higher tendency of sarcastic comments. US political party like "GOP" or "Republican" and "DNC" or "Democrats" have balanced sarcasm rate and it may be due to the fact that supporters for each party (cover almost 50% of the voter population for each party) leaving less sarcastic comments and as a result balancing the sarcasm rate.

## Text Mining and Sentiment Analysis


## Data Splitting and Sub-Sampling


## Select Modeling Techniques


## Build the Models


## Assess the Models

## Conclusion and Discussion
We see that there is a considerable amount of sarcastic comments on different reddit threads. The ability to detect sarcasm in any text will help summarizing the context of the text better. One application of the sarcasm detector is to help out people (and the social media platform) with better information. In this way, the said social media platform can restrict that post, comment or thread from reaching a lot of people. Sarcastic remarks are often harmful leading to dire consequences. The sarcasm detector should have a positive influence on understanding the way that we do natural language understanding. 
Following are few potential implications of a sarcasm detector:
1) Improved communication: A sarcasm detector should be help individuals and organizations get better at communicating. People can understand when someone is being sarcastic and therefore could lead to improved communication and fewer misunderstandings.
2) Increased productivity: Sarcasm can be a source of tension or conflict at work. Therefore detecting sarcasm could help reduce tension and improve productivity.
3) Ethical concerns: There are ethical concerns associated with developing and implementing a sarcasm detector. For example, some people may view it as an invasion of privacy or as a way to police speech. There could also be concerns around false positives (i.e., the detector identifying sarcasm where there is none) and false negatives (i.e., the detector failing to identify sarcasm when it is present).
4) Impact on humour: Sarcasm is often used in humour, and a sarcasm detector could have implications for the use of sarcasm in comedy and entertainment. 
5) Marketing: A very common issue that marketeers face is understanding whether the customers actually liked the product or they are being sarcastic in the review that they have left especially on social media platforms. A robust sarcasm detector can help with that kind of targeted marketing as the marketeers will have a better understanding of the review and they can suggest changes to the product and mend their ways in which they target people. 

It is very important to develop a robust sarcasm detector which identifies different situations in which sarcasm is used but not police the speech so much that it is an ethical dilema to deploy the model. 


## References

[1] Wong, Q.(2022, July 27). Hiding Hate in Plain Sight: How Social Media Sites Misread Sarcasm. *CNET.* https://www.cnet.com/news/social-media/hiding-hate-in-plain-sight-how-social-media-sites-misread-sarcasm/

[2] Peters, S. (2018, March 8). Why is sarcasm so difficult to detect in texts and emails? *The Conversation.* https://theconversation.com/why-is-sarcasm-so-difficult-to-detect-in-texts-and-emails-91892

[3] Khodak, M., Saunshi, N., & Vodrahalli, K. (2018, March 22). A Large Self-Annotated Corpus for Sarcasm. *arXiv preprint arXiv:1704.05579 [Computation and Language].* https://doi.org/10.48550/arXiv.1704.05579   

[4] Ofer, D. (2018, May 27). Sarcasm on Reddit [Dataset]. *Kaggle.* https://www.kaggle.com/datasets/danofer/sarcasm
