import requests
from bs4 import BeautifulSoup
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Function to scrape course data from the website
def scrape_courses(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    course_elements = soup.find_all("div", class_="course")
    courses = []
    for course in course_elements:
        title = course.find("h2").text
        description = course.find("p").text
        course_url = course.find("a")["href"]
        courses.append({"title": title, "description": description, "url": course_url})
    return courses

# Function to generate a personalized course recommendation
def generate_personalized_recommendation(user_profile, courses):
    import random
    return random.choice(courses)

# Load pre-trained GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Define the website URL for course data
course_url = "https://brainlox.com/courses/category/technical"

# Scrape course data
courses = scrape_courses(course_url)

# Create user profiles (dictionary of user preferences)
user_profiles = {
    "user1": {"interests": ["programming", "web development"], "level": "intermediate"},
    "user2": {"interests": ["data science", "machine learning"], "level": "advanced"},
}

# Chatbot introduction
print("Chatbot: Hi, I'm your personalized course recommendation chatbot.")
print("Chatbot: You can ask for course recommendations, filter courses, or type 'exit' to quit.")

while True:
    user_input = input("You: ").strip().lower()
    
    if user_input == "exit":
        print("Chatbot: Goodbye!")
        break
    
    if "recommend" in user_input:
        # Determine the user (for simplicity, let's assume 'user1' for this example)
        current_user = "user1"
        
        # Generate a personalized course recommendation based on user input
        recommendation = generate_personalized_recommendation(user_profiles[current_user], courses)
        
        # Generate a response using GPT-2
        response = model.generate(
            tokenizer.encode(f"Chatbot: I recommend the course '{recommendation['title']}' because {recommendation['description']}", return_tensors="pt"),
            max_length=100,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
        )
        
        print("Chatbot:", tokenizer.decode(response[0], skip_special_tokens=True))
    
    elif "courses" in user_input:
        # Provide options for filtering courses
        print("Chatbot: You can filter courses by interest or level. Please specify your filter, e.g., 'Filter by interest: programming'.")
    
    elif "filter by interest" in user_input:
        # Filter courses by user's interests (for simplicity, let's assume 'programming' for this example)
        filtered_courses = [course for course in courses if "programming" in course['description'].lower()]
        if filtered_courses:
            print("Chatbot: Here are some programming courses:")
            for index, course in enumerate(filtered_courses[:5], start=1):
                print(f"{index}. {course['title']}")
        else:
            print("Chatbot: Sorry, no programming courses found.")
    
    elif "filter by level" in user_input:
        # Filter courses by user's level (for simplicity, let's assume 'intermediate' for this example)
        filtered_courses = [course for course in courses if "intermediate" in course['description'].lower()]
        if filtered_courses:
            print("Chatbot: Here are some intermediate level courses:")
            for index, course in enumerate(filtered_courses[:5], start=1):
                print(f"{index}. {course['title']}")
        else:
            print("Chatbot: Sorry, no intermediate level courses found.")
    
    else:
        # Handle unrecognized input
        print("Chatbot: I'm not sure how to respond to that. You can ask for course recommendations, filter courses, or type 'exit' to quit.")

