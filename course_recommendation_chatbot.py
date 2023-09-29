# Install necessary libraries
!pip install beautifulsoup4
!pip install requests
!pip install transformers
!pip install scikit-learn
!pip install spacy
!python -m spacy download en_core_web_sm
!python -m spacy download es_core_news_sm
!python -m spacy download fr_core_news_sm
!python -m spacy download de_core_news_sm

# Import necessary libraries
import requests
from bs4 import BeautifulSoup
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import logging
import spacy

# Redirect Transformers library output to null device
logging.getLogger("transformers").setLevel(logging.ERROR)

# Load language models
nlp_en = spacy.load("en_core_web_sm")
nlp_es = spacy.load("es_core_news_sm")
nlp_fr = spacy.load("fr_core_news_sm")
nlp_de = spacy.load("de_core_news_sm")

# Define a function to scrape course data
def scrape_course_data(url):
    # Send a GET request to the URL
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML content of the page
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find all script elements
        script_elements = soup.find_all('script')

        # Search for the specific content block in the scripts
        target_content = None
        for script in script_elements:
            content = script.string
            if content and 'props":{"pageProps":{"courses":[{"id":"' in content:
                target_content = content
                break

        # If the target content is found, parse it as JSON
        if target_content:
            data = json.loads(target_content)

            # Extract and store the 'title' and 'overview' in a list
            courses = data.get('props', {}).get('pageProps', {}).get('courses', [])
            course_data = []

            for course in courses:
                title = course.get('title')
                overview = course.get('overview')

                if title and overview:
                    course_data.append({'title': title, 'overview': overview})

            return course_data
        else:
            return None
    else:
        print(f'Failed to retrieve the webpage. Status code: {response.status_code}')
        return None

# Define a function to recommend a course based on user input
def recommend_course(user_input, courses):
    # Define a TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer()

    # Create TF-IDF matrices for user input and course overviews
    tfidf_matrix = tfidf_vectorizer.fit_transform([user_input] + [course['overview'] for course in courses])

    # Calculate cosine similarities between user input and course overviews
    cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    # Find the index of the course with the highest similarity
    best_match_index = cosine_similarities.argmax()

    # Recommend the course with the highest similarity
    recommended_course = courses[best_match_index]

    return recommended_course

# Load the courses data from the specified URL
courses_data = scrape_course_data('https://brainlox.com/courses/category/technical')

# Load pre-trained GPT-2 model and tokenizer
model_name = "gpt2"  # You can use other GPT-2 variants like gpt2-medium, gpt2-large, etc.
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Define a function to generate responses using GPT-2
def generate_response(prompt, max_length=50):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long)  # Set attention mask to 1
    response_ids = model.generate(input_ids, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2, attention_mask=attention_mask, pad_token_id=model.config.pad_token_id)
    response = tokenizer.decode(response_ids[0], skip_special_tokens=True)
    return response

# Main chat loop
print("Chatbot: Hello! How can I assist you today?")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Chatbot: Goodbye!")
        break

    # Detect the language of user input
    doc = nlp_en(user_input)  # Default to English if the language is not recognized
    detected_language = doc.lang_

    # Process user input based on detected language
    if detected_language == "es":
        # User input is in Spanish
        # Implement logic for handling Spanish input here
        pass
    elif detected_language == "fr":
        # User input is in French
        # Implement logic for handling French input here
        pass
    elif detected_language == "de":
        # User input is in German
        # Implement logic for handling German input here
        pass
    else:
        # User input is in English or an unsupported language
        # Implement logic for handling English or other languages here
        pass

    # Recommend a course based on user input
    recommended_course = recommend_course(user_input, courses_data)

    # Generate a chatbot response
    chatbot_response = generate_response(user_input)

    print("Chatbot:", chatbot_response)
    print("Recommended Course:", recommended_course["title"])
