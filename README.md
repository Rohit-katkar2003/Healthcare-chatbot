Healthcare Chatbot

This project is a healthcare chatbot designed to provide patients with health-related queries and consultations. The chatbot utilizes a fine-tuned Flan-T5-base model on domain-specific data, enabling efficient and accurate responses. The project is optimized to work seamlessly on CPU, making it accessible and cost-effective.

Features

Health-related Query Assistance: Provides accurate responses to medical queries.

Consultation Support: Assists patients with personalized healthcare advice.

Fine-Tuned Model: Flan-T5-base model fine-tuned on healthcare-specific data for improved performance.

CPU-Friendly: Designed to work efficiently on systems without GPUs.

Project Structure

The project follows the structure below:
'''bash
Healthcare-Chatbot/
├── src/
│   ├── __init__.py        # Package initialization
│   ├── helper.py          # Helper functions
│   └── prompt.py          # Prompt generation logic
├── .env                   # Environment variables
├── setup.py               # Setup script
├── app.py                 # Main application logic
├── research/
│   └── trials.ipynb       # Jupyter notebook for research and experimentation
├── store_index.py         # Indexing logic
├── static/                # Static files (CSS, JavaScript, images)
├── templates/
│   └── chat.html          # HTML template for the chat interface
'''
Getting Started

1. Download the Project

Clone the repository to your local machine:
'''bash
git clone https://github.com/Rohit-katkar2003/Healthcare-chatbot.git
cd Healthcare-chatbot
'''
2. Create and Activate a Virtual Environment

Create a virtual environment named Chatbot:
'''bash
python -m venv Chatbot
'''
Activate the environment:

Windows:
'''bash
Chatbot\Scripts\activate
'''

3. Install Dependencies

Install the required dependencies:
'''bash
pip install -r requirements.txt
'''
4. Run the Application

Start the chatbot application:
'''bash
python app.py
'''
Fine-Tuned Model

The chatbot uses a fine-tuned Flan-T5-base model. The model was fine-tuned on a healthcare-specific dataset to ensure accurate and reliable responses. This allows the chatbot to:

Understand complex healthcare-related questions.

Provide precise and context-aware answers.

Usage

Navigate to the local server URL displayed after running the application (usually http://127.0.0.1:5000).

Enter your query in the chat interface.

Receive a response tailored to your healthcare question.

Future Enhancements

Integration with real-time medical databases for updated responses.

Multi-language support for diverse user bases.

Mobile application version for ease of access.

Contributing

Contributions are welcome! Follow these steps to contribute:

Fork the repository.

Create a new branch:
'''bash
git checkout -b feature-branch
'''
Commit your changes:
'''bash
git commit -m "Add new feature"
'''
Push to the branch:
'''bash
git push origin feature-branch
'''
Open a pull request.

License

This project is licensed under the MIT License.

Contact

For any queries or support, feel free to contact:
Rohit B. KatkarGitHub Profile