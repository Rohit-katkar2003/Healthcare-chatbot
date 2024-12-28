```bash
# **HEALTHCARE CHATBOT**

## **🚀 Project Overview**
This project is a **healthcare chatbot** designed to provide patients with health-related queries and consultations. By leveraging a fine-tuned **Flan-T5-base** model, the chatbot delivers domain-specific responses with high efficiency and accuracy. 

This solution is optimized to run seamlessly on **CPU**, making it cost-effective and accessible to a wider audience without requiring GPU resources.

## **✨ Features**
- **🩺 Health-related Query Assistance:** Provides instant and accurate responses to medical questions.
- **💡 Consultation Support:** Assists patients with personalized healthcare advice based on their queries.
- **🔧 Fine-Tuned Model:** Utilizes a Flan-T5-base model fine-tuned on healthcare-specific data for enhanced performance.
- **💻 CPU-Friendly:** Designed to operate efficiently without the need for GPU resources, ensuring broader accessibility.

---

## **📂 Project Structure**
The project adheres to the following structure:

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

---

## **⚙️ Getting Started**

### **1️⃣ Download the Project**
Clone the repository to your local machine:
git clone https://github.com/Rohit-katkar2003/Healthcare-chatbot.git
cd Healthcare-chatbot

### **2️⃣ Create and Activate a Virtual Environment**
Create a virtual environment named `Chatbot`:
python -m venv Chatbot

Activate the environment:
- **Windows:**
  Chatbot\Scripts\activate
- **Mac/Linux:**
  source Chatbot/bin/activate

### **3️⃣ Install Dependencies**
Install the required dependencies:
pip install -r requirements.txt

### **4️⃣ Run the Application**
Start the chatbot application:
python app.py

---

## **🤖 Fine-Tuned Model**
The chatbot utilizes a fine-tuned **Flan-T5-base** model, specifically trained on healthcare-related data. This allows the chatbot to:
- 🧠 Understand and interpret complex medical queries.
- 🏥 Provide precise, context-aware answers for patient inquiries.
- 📊 Enhance accuracy and relevance in healthcare consultations.

---

## **🛠️ Usage**
1. Launch the chatbot by running the app.
2. Open the local server URL (typically `http://127.0.0.1:5000`).
3. Input health-related queries through the chat interface.
4. Receive personalized and relevant responses instantly.

---

## **🔮 Future Enhancements**
- 🌐 **Real-time Medical Data Integration:** Connect to live medical databases for up-to-date information.
- 🗣️ **Multi-language Support:** Enable chatbot functionality in various languages.
- 📱 **Mobile Application:** Develop a mobile-friendly version for greater accessibility.

---

## **🤝 Contributing**
Contributions are always welcome! Follow these steps to contribute:
1. **Fork** the repository.
2. Create a new branch:
git checkout -b feature-branch
3. **Commit** your changes:
git commit -m "Add new feature"
4. **Push** to the branch:
git push origin feature-branch
5. Open a **pull request**.

---

## **📜 License**
This project is licensed under the [MIT License](LICENSE).

---

## **📧 Contact**
For any queries or support, feel free to reach out:
**👨‍💻 Rohit B. Katkar**  
🔗 [GitHub Profile](https://github.com/Rohit-katkar2003)
```

