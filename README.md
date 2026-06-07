# 🔬 AI-Powered Steel Defect Inspection and Root Cause Analysis System

[![Live Demo](https://img.shields.io/badge/HuggingFace-Live%20Demo-yellow)](https://huggingface.co/spaces/bhuviiiiiiiii/steel-defect-inspector)

An intelligent steel quality inspection system that combines **Deep Learning**, **Explainable AI**, and **Metallurgical Domain Knowledge** to automatically detect steel surface defects, estimate defect severity, perform root cause analysis, assess industrial risk, and provide corrective action recommendations.

---

## 🚀 Live Demo

### Hugging Face Space

👉 https://huggingface.co/spaces/bhuviiiiiiiii/steel-defect-inspector

Upload a steel surface image and receive:

* Defect Classification
* GradCAM Visualization
* Defect Area Estimation
* Severity Assessment
* Root Cause Analysis
* RPN-Based Risk Scoring
* Corrective Action Recommendations

---

## 📌 Industrial Significance

Steel surface defects can lead to product rejection, increased production costs, reduced customer satisfaction, and process inefficiencies.

Traditional AI systems typically stop at defect classification. This project extends beyond detection by providing explainable diagnostics and actionable insights that can support quality engineers in identifying probable process issues and implementing corrective measures.

---

## 🎯 Key Contributions

* EfficientNet-B3 based steel defect classification
* GradCAM-based explainable AI visualization
* Defect area-based severity estimation
* Metallurgical root cause analysis
* Process parameter inspection guidance
* RPN-based industrial risk assessment
* Corrective action recommendation system
* Interactive Gradio web deployment

---

## 🔄 System Workflow

```text
Steel Surface Image
        ↓
EfficientNet-B3
        ↓
Defect Classification
        ↓
GradCAM Visualization
        ↓
Defect Area Estimation
        ↓
Severity Assessment
        ↓
Root Cause Analysis
        ↓
RPN Risk Assessment
        ↓
Corrective Action Recommendations
```

---

## 🧠 Model Information

| Parameter             | Value                            |
| --------------------- | -------------------------------- |
| Architecture          | EfficientNet-B3                  |
| Dataset               | NEU Steel Surface Defect Dataset |
| Number of Classes     | 6                                |
| Explainability Method | GradCAM                          |
| Severity Assessment   | Defect Area Based                |
| Risk Assessment       | RPN Based                        |
| Deployment            | Gradio + Hugging Face Spaces     |

---

## 🗂️ Defect Classes

| Defect Class    | Description                                           |
| --------------- | ----------------------------------------------------- |
| Crazing         | Fine crack network caused by thermal stresses         |
| Inclusion       | Non-metallic particles embedded in steel              |
| Patches         | Irregular scale or discoloration defects              |
| Pitted Surface  | Surface craters caused by corrosion or casting issues |
| Rolled-In Scale | Oxide scale folded into steel during rolling          |
| Scratches       | Linear marks caused by mechanical contact             |

---

## 📊 Dataset

### NEU Steel Surface Defect Dataset (NEU-DET)

* 1800 steel surface images
* 6 defect categories
* Industrial steel surface defects
* Widely used benchmark dataset for defect inspection research

Dataset Source:

https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database

---

## 🔍 Technical Highlights

### GradCAM Explainability

The system highlights image regions influencing model predictions, improving transparency and interpretability of AI decisions.

### Defect Severity Assessment

Unlike conventional systems that rely solely on prediction confidence, severity is estimated using the activated defect area identified through GradCAM visualization.

### Metallurgical Root Cause Analysis

Each detected defect is mapped to probable process-level causes such as:

* Inadequate descaling
* Improper furnace conditions
* Reoxidation during casting
* Mechanical handling issues
* Cooling-related thermal stresses

### Corrective Action Recommendations

The system provides process-oriented recommendations to assist quality engineers in reducing defect occurrence and improving process control.

### RPN Risk Assessment

Defects are prioritized using Risk Priority Numbers (RPN), inspired by industrial Failure Mode and Effects Analysis (FMEA) methodologies.

---

## 📄 Sample Output

The generated inspection report includes:

* Detected Defect
* Confidence Score
* Defect Area Percentage
* Severity Assessment
* Risk Level (RPN)
* Root Cause Analysis
* Process Parameters to Check
* Corrective Actions

---

## 📁 Repository Structure

```text
ai-steel-defect-inspection-system/
│
├── app.py
├── requirements.txt
├── steel-defect-gradio-demo.ipynb
├── self-project-mmed.ipynb
├── sample_report.pdf
└── README.md
```

---

## 🛠️ Installation

Clone the repository:

```bash
git clone https://github.com/baheti777/ai-steel-defect-inspection-system.git

cd ai-steel-defect-inspection-system
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Place the trained model file (`best_model.pth`) in the project directory and run:

```bash
python app.py
```

---

## 🏭 Applications

* Steel Manufacturing Plants
* Rolling Mills
* Quality Assurance Departments
* Smart Manufacturing Systems
* Industrial Inspection Automation
* Industry 4.0 Quality Control

---

## 👨‍💻 Author

**Bhuvi Baheti**

B.Tech(3rd year), Metallurgical and Material Engineering <br>
Indian Institute Of Technology, Roorkee

---

## ⭐ Future Improvements

* Multi-defect detection
* Real-time video inspection
* Object detection-based localization
* Defect trend analytics dashboard
* Integration with Manufacturing Execution Systems (MES)

---

