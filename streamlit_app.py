import tempfile
import os
import dspy
import streamlit as st
from dspy import (
    Signature,
    Module,
    InputField,
    OutputField,
    Predict,
    Prediction,
    Example,
)
from dspy.teleprompt import BootstrapFewShot
from llama_parse import LlamaParse
from dspy.evaluate import Evaluate
from dsp import LM


LLAMA_INDEX_KEY = st.secrets["LLAMA_INDEX_KEY"]
CLAUDE_KEY = st.secrets["CLAUDE_KEY"]


class Claude(LM):
    def __init__(self, model, api_key, **kwargs):
        super().__init__(**self.kwargs)
        self.model = model
        self.api_key = api_key
        self.provider = "default"
        self.base_url = "https://api.anthropic.com/v1/messages"

    def basic_request(self, prompt: str, **kwargs):
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "messages-2023-12-15",
            "content-type": "application/json",
        }

        data = {
            **kwargs,
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }

        response = requests.post(self.base_url, headers=headers, json=data)
        response = response.json()

        self.history.append(
            {
                "prompt": prompt,
                "response": response,
                "kwargs": kwargs,
            }
        )
        return response

    def __call__(self, prompt, only_completed=True, return_sorted=False, **kwargs):
        response = self.request(prompt, **kwargs)

        completions = [result["text"] for result in response["content"]]

        return completions


sonnet = Claude("claude-3-5-sonnet-20240620", CLAUDE_KEY)
dspy.settings.configure(lm=sonnet)


@st.cache_data
def parse_local_pdf(filename):
    try:
        parser = LlamaParse(
            api_key=LLAMA_INDEX_KEY,
            result_type="text",
            verbose=True,
        )
        doc = parser.load_data(filename)
        return doc[0].text
    except Exception as e:
        st.error(f"Error parsing resume with LlamaParse: {e}")
        return None


def parse_input_pdf(label):
    uploaded_file = st.file_uploader(label)
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        with tempfile.NamedTemporaryFile(suffix=".pdf") as tf:
            tf.write(bytes_data)
            tf.flush()
            return parse_local_pdf(tf.name)


class GenerateInterviewQuestion(dspy.Signature):
    """Based on resume, job description, and last answer."""

    resume = dspy.InputField(
        desc="Text of the candidate's resume detailing their skills and experience"
    )
    job = dspy.InputField(
        desc="Text of the job description outlining required qualifications"
    )
    last_answer = dspy.InputField(
        desc="Text of the candidate's last answer", required=False
    )  # Make this optional if you want to start without an initial answer
    question = dspy.OutputField(
        desc="Interview question that relates to the candidate's suitability for the job based on their resume, job description, and last answer."
    )
    print("Signature updated with last answer...")


class InterviewQuestionGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_question = Predict(GenerateInterviewQuestion)

    def forward(self, resume, job, last_answer=None):
        prediction = self.generate_question(
            resume=resume, job=job, last_answer=last_answer
        )
        return dspy.Prediction(question=prediction.question)

    print("Module updated to handle last answer...")


# Training Examples and Optimization

# Define examples with the necessary fields


# def main():
training_resume_text = parse_local_pdf("trainingdata/resume.pdf")
training_job_text = parse_local_pdf("trainingdata/job.pdf")
from trainingdata import training_texts

# return resume_text, job_text


# if __name__ == "__main__":
#    resume_text, job_text = main()

examples = []
previous_questions = ""
last_answer = ""
for i, (question, answer) in enumerate(training_texts):
    examples.append(
        dspy.Example(
            resume=training_resume_text,
            job=training_job_text,
            last_answer=last_answer,
            previous_questions=previous_questions,
            question=question,
        )
    )
    last_answer = answer
    previous_questions += question

print("Training examples created...")

trainset = [ex.with_inputs("resume", "job", "last_answer") for ex in examples]

print("Trainset created...")


skill_keywords = {
    "code": ["java", "python", "go", "code", "programming"],
    "cloud": ["aws", "gcp", "azure"],
}


def identify_current_skill(question, previous_questions):
    question_text = question.lower()
    for skill, keywords in skill_keywords.items():
        if any(
            keyword in question_text
            or any(keyword in q.lower() for q in previous_questions)
            for keyword in keywords
        ):
            return skill
    return None


class Assess(dspy.Signature):
    """Assesses the interview question for question count within a skill."""

    assessed_text = dspy.InputField()  # Interview question
    assessment_question = dspy.InputField()  # Question for LLM assessment
    assessment_answer = dspy.OutputField(desc="Yes or No")  # LLM's answer


def metric(gold, pred, trace=None):
    # Identify current skill based on previous questions
    question = pred.question.lower()
    previous_questions = gold.previous_questions
    current_skill = identify_current_skill(question, previous_questions)
    with dspy.context(lm=sonnet):
        count_check = dspy.Predict(Assess)(
            assessed_text=question,
            assessment_question=f"Have there already been 3 questions asked about the same skill ({current_skill}) as this question?",
        )
    too_many = count_check.lower() == "yes"
    score = 0 if too_many else 1
    return 1 if trace else score


print("Metric created...")


class FinalQuestionGenerator:
    def __init__(self, resume, job, metric, trainset):
        evaluator = Evaluate(
            devset=trainset, num_threads=5, display_progress=True, display_table=11
        )
        evaluation_score = evaluator(InterviewQuestionGenerator(), metric)
        print("Evaluation is good...")
        print(f"Average Metric: {evaluation_score}")
        teleprompter = BootstrapFewShot(
            metric=metric, max_bootstrapped_demos=5, max_labeled_demos=5
        )
        self.module = teleprompter.compile(
            InterviewQuestionGenerator(), trainset=trainset
        )
        print("Training completed...")

    def gen_question(self, resume, job, last_answer=None):
        pred = self.module(resume=resume, job=job, last_answer=last_answer)
        return pred.question


final = FinalQuestionGenerator(
    training_resume_text, training_job_text, metric, trainset
)

# Streamlit App
st.title("DSPy-Optimized AI Interview Assistant")
resume_text = parse_input_pdf("Resume (PDF Only)")
job_text = parse_input_pdf("Job Description (PDF Only)")
generated_questions = []
generated_question_1 = None


generated_question = None
user_answer = None

if st.button("Generate Interview Question"):
    if not (resume_text and job_text):
        st.error("Please upload both your resume and job description!")
    else:
        previous_question = None
        while 1:
            question = final.generate_question(
                resume=resume_text, job=job_text, last_answer=previous
            )
            if question:
                st.write(question)
                answer = st.text_area("Your Answer here:")
                submit_answer = st.button("Submit Your Answer")
                if answer:
                    if answer == "bye":
                        break
                    # TODO: feedback
                    st.success("Answer submitted successfully!")
                else:
                    st.warning("Please provide an answer before proceeding.")
            else:
                break
            previous_question = question
