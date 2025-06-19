from llama_api_client import LlamaAPIClient
import os
client = LlamaAPIClient(
    api_key=os.environ["LLAMA_API_KEY"],
)

tools = [{
    'type': 'function', 
    'function': {
        'name': 'submit_review', 
        'description': 'Submit a review evaluating the output of the training script.', 
        'parameters': {
            'type': 'object', 
            'properties': {
                'is_bug': {'type': 'boolean', 'description': 'true if the output log shows that the execution failed or has some bug, otherwise false.'}, 
                'summary': {'type': 'string', 'description': 'if there is a bug, propose a fix. Otherwise, write a short summary (2-3 sentences) describing the empirical findings.'}, 
                'metric': {'type': 'number', 'description': 'If the code ran successfully, report the value of the validation metric. Otherwise, leave it null.'}, 'lower_is_better': {'type': 'boolean', 'description': 'true if the metric should be minimized (i.e. a lower metric value is better, such as with MSE), false if the metric should be maximized (i.e. a higher metric value is better, such as with accuracy).'}
                }, 
            'required': ['is_bug', 'summary', 'metric', 'lower_is_better']}}}]

tool_choice = {
			"type": "function", 
			"function": {"name": 'submit_review'},
		}

# tools = [{
#   'type': 'function',
#   'function': {
#     'name': 'web_search',
#     'description': 'Search the web for information',
#     'parameters': {
#       'properties': {
#         'query': {
#           'type': 'string',
#           'description': 'The query to search for'
#         }
#       },
#       'required': ['query']
#     }
#   }
# }]


completion = client.chat.completions.create(
    model="Llama-4-Maverick-17B-128E-Instruct-FP8",
    messages=[{'role': 'system', 'content': '# Introduction\n\nYou are a data scientist attending a competition. You have written code to solve this task and now need to evaluate the output of the code execution. You should determine if there were any bugs as well as report the empirical findings.\n\n# Task description\n\n# Machine Learning Engineer Challenge: TimeSeriesGym Competition\n\n## Your Role\nYou are a Machine Learning Engineer tasked with solving complex ML problems in the TimeSeriesGym offline competition. This competition tests your ability to understand problems, follow instructions, and build/train/evaluate ML models.\n\n## Core Requirements\n- You must solve each task by training a model and generating predictions on the test set\n- Hand-labeling the data to produce submissions is strictly prohibited\n\n## Environment Setup\n\n### Critical Files and Locations\n- **Competition Instructions**: Located at `data/description.md` - This is your primary reference for understanding the specific problem\n- **Dataset**: All data files are in `data/` directory\n- **Submission Path**: Your final submission MUST be saved to `submission/` in the format specified in the instructions\n- **Environment Instructions**: These general instructions can be found at `instructions.txt`\n\n### Test Dataset Information\n- We\'ve created our own train/test splits for evaluation purposes\n- The data in `data/` may differ slightly from what\'s described in the competition instructions\n- Use the provided train/test splits for your work\n\n## Submission Instructions\n\n### File Format and Location\n- Your submission MUST be saved to the exact path: `submission/`\n- The file format must strictly follow the specifications in `description.md`\n- Failing to use the correct path or format will result in a failed evaluation\n\n### Validation Process\n1. Before finalizing your submission, validate its format using the validation server\n2. Use the command: `curl -X POST -F "file=@${SUBMISSION_FILE}" http://localhost:5000/validate`\n3. Alternatively, use the provided script: `validate_submission.sh`\n4. The validator will confirm if your submission format is valid (but won\'t provide a score)\n5. Make corrections if needed and revalidate until you receive a confirmation\n\n### Final Submission\n- Once validated, ensure your final submission file remains at `submission/`\n- Only the last file in this location will be evaluated\n- Double-check that your model\'s predictions cover the entire test set\n- Review the submission format requirements one last time before completing\n\n## Evaluation Criteria\n- Your submission will be graded based on the metric specified in the competition instructions\n- Your goal is to achieve the highest possible score\n\n## Important Rules\n- Viewing or copying others\' solutions is prohibited and will result in disqualification\n- In case of conflicts between these instructions and `data/description.md`, these general instructions take priority\n\nADDITIONAL NOTES\n------\n\n- **Compute**: You have access to 2 NVIDIA A100-SXM4-80GB with the appropriate drivers installed.\n- **Total Runtime**: You have a maximum of 1hrs 0mins 0secs hours to attempt this task. You may iteratively develop your solution within this time frame, but bear in mind that program runtime also counts towards this limit.\n- **Total Steps**: You have a maximum of 50 steps to submit your solution.\n- You will be informed of the remaining time and steps at each step.\n- **Use as much of the available time as possible to refine and optimize your submission**: Your goal is to achieve the best possible score, not just to submit your first working solution. Consider strategies like validating your answer using the "Validate Submission" tool at the http://localhost:5000/validate endpoint, manually spot-checking predictions, building custom validation sets and grading functions, and comparing different algorithms. You may keep working until the time limit expires -- as long as your best submission is stored at the designated location at the end of your run, it will be picked up for grading.\n\nCOMPETITION INSTRUCTIONS\n------\n\n# Optiver - Trading at the Close\n\n**Competition Link:** https://www.kaggle.com/competitions/optiver-trading-at-the-close/overview\n\n## Overview\n\nIn this competition, you are challenged to develop a model capable of predicting the closing price movements for hundreds of Nasdaq listed stocks using data from the order book and the closing auction of the stock. Information from the auction can be used to adjust prices, assess supply and demand dynamics, and identify trading opportunities.\n\n### Background\n\nStock exchanges are fast-paced, high-stakes environments where every second counts. The intensity escalates as the trading day approaches its end, peaking in the critical final ten minutes. These moments, often characterised by heightened volatility and rapid price fluctuations, play a pivotal role in shaping the global economic narrative for the day.\n\nEach trading day on the Nasdaq Stock Exchange concludes with the Nasdaq Closing Cross auction. This process establishes the official closing prices for securities listed on the exchange. These closing prices serve as key indicators for investors, analysts and other market participants in evaluating the performance of individual securities and the market as a whole.\n\nWithin this complex financial landscape operates Optiver, a leading global electronic market maker. Fueled by technological innovation, Optiver trades a vast array of financial instruments, such as derivatives, cash equities, ETFs, bonds, and foreign currencies, offering competitive, two-sided prices for thousands of these instruments on major exchanges worldwide.\n\nIn the last ten minutes of the Nasdaq exchange trading session, market makers like Optiver merge traditional order book data with auction book data. This ability to consolidate information from both sources is critical for providing the best prices to all market participants.\n\nYour model can contribute to the consolidation of signals from the auction and order book, leading to improved market efficiency and accessibility, particularly during the intense final ten minutes of trading. You\'ll also get firsthand experience in handling real-world data science problems, similar to those faced by traders, quantitative researchers and engineers at Optiver.\n\n### Evaluation\n\nSubmissions are evaluated on the **Mean Absolute Error (MAE)** between the predicted return and the observed target.\n\nWhere:\n- n is the total number of data points\n- y_i is the predicted value for data point i  \n- x_i is the observed value for data point i\n\n### Submission Requirements\n\nYou must submit to this competition using the provided python time-series API, which ensures that models do not peek forward in time.\n\n**Technical Requirements:**\n- CPU Notebook ≤ 9 hours run-time\n- GPU Notebook ≤ 9 hours run-time\n- Internet access disabled\n- Freely & publicly available external data is allowed, including pre-trained models\n- Submission file must be named submission.csv and be generated by the API\n\n### Prizes\n\n- **1st Place:** $25,000\n- **2nd Place:** $20,000\n- **3rd Place:** $15,000\n- **4th Place:** $10,000\n- **5th - 10th Place:** $5,000 each\n\n## Dataset Description\n\nThis dataset contains historic data for the daily ten minute closing auction on the NASDAQ stock exchange. Your challenge is to predict the future price movements of stocks relative to the price future price movement of a synthetic index composed of NASDAQ-listed stocks.\n\nThis is a forecasting competition using the time series API. The private leaderboard will be determined using real market data gathered after the submission period closes.\n\n### Files\n\n**[train/test].csv** - The auction data. The test data will be delivered by the API.\n\n**Features:**\n- **stock_id** - A unique identifier for the stock. Not all stock IDs exist in every time bucket\n- **date_id** - A unique identifier for the date. Date IDs are sequential & consistent across all stocks\n- **imbalance_size** - The amount unmatched at the current reference price (in USD)\n- **imbalance_buy_sell_flag** - An indicator reflecting the direction of auction imbalance:\n  - 1: buy-side imbalance\n  - -1: sell-side imbalance  \n  - 0: no imbalance\n- **reference_price** - The price at which paired shares are maximized, the imbalance is minimized and the distance from the bid-ask midpoint is minimized, in that order. Can also be thought of as being equal to the near price bounded between the best bid and ask price\n- **matched_size** - The amount that can be matched at the current reference price (in USD)\n- **far_price** - The crossing price that will maximize the number of shares matched based on auction interest only. This calculation excludes continuous market orders\n- **near_price** - The crossing price that will maximize the number of shares matched based auction and continuous market orders\n- **[bid/ask]_price** - Price of the most competitive buy/sell level in the non-auction book\n- **[bid/ask]_size** - The dollar notional amount on the most competitive buy/sell level in the non-auction book\n- **wap** - The weighted average price in the non-auction book\n- **seconds_in_bucket** - The number of seconds elapsed since the beginning of the day\'s closing auction, always starting from 0\n- **target** - The 60 second future move in the wap of the stock, less the 60 second future move of the synthetic index. Only provided for the train set\n\n**Additional Files:**\n- **sample_submission** - A valid sample submission, delivered by the API\n- **revealed_targets** - When the first time_id for each date (i.e. when seconds_in_bucket equals zero) the API will serve a dataframe providing the true target values for the entire previous date\n- **public_timeseries_testing_util.py** - An optional file intended to make it easier to run custom offline API tests\n- **example_test_files/** - Data intended to illustrate how the API functions\n- **optiver2023/** - Files that enable the API\n\n### Important Notes\n\n- The synthetic index is a custom weighted index of Nasdaq-listed stocks constructed by Optiver for this competition\n- The unit of the target is basis points, which is a common unit of measurement in financial markets. A 1 basis point price move is equivalent to a 0.01% price move\n- All size related columns are in USD terms\n- All price related columns are converted to a price move relative to the stock wap (weighted average price) at the beginning of the auction period\n- The API is expected to deliver all rows in under five minutes and to reserve less than 0.5 GB of memory\n- The first three date ids delivered by the API are repeats of the last three date ids in the train set for illustration purposes\n\n# Implementation\n\n```python\nimport pandas as pd\nimport numpy as np\nfrom sklearn.preprocessing import MinMaxScaler\nfrom keras.models import Sequential\nfrom keras.layers import LSTM, Dense\n\n# Load the data\ntrain_df = pd.read_csv("input/train.csv")\ntest_df = pd.read_csv("input/test.csv")\n\n# Scale the data\nscaler = MinMaxScaler()\ntrain_df[\n    ["ask_price", "bid_price", "far_price", "near_price", "reference_price", "wap"]\n] = scaler.fit_transform(\n    train_df[\n        ["ask_price", "bid_price", "far_price", "near_price", "reference_price", "wap"]\n    ]\n)\ntest_df[\n    ["ask_price", "bid_price", "far_price", "near_price", "reference_price", "wap"]\n] = scaler.fit_transform(\n    test_df[\n        ["ask_price", "bid_price", "far_price", "near_price", "reference_price", "wap"]\n    ]\n)\n\n# Split the data into input and output\nX_train = train_df.drop(["target"], axis=1)\ny_train = train_df["target"]\nX_test = test_df.drop(["target"], axis=1)\n\n# Create the LSTM model\nmodel = Sequential()\nmodel.add(LSTM(50, input_shape=(X_train.shape[1], 1)))\nmodel.add(Dense(1))\nmodel.compile(loss="mean_absolute_error", optimizer="adam")\n\n# Reshape the data for the LSTM model\nX_train = X_train.values.reshape(-1, X_train.shape[1], 1)\nX_test = X_test.values.reshape(-1, X_test.shape[1], 1)\n\n# Train the model\nmodel.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)\n\n# Make predictions on the test data\npredictions = model.predict(X_test)\n\n# Save the predictions to a submission file\nsubmission_df = pd.DataFrame(predictions, columns=["target"])\nsubmission_df.to_csv("submission/submission.csv", index=False)\n\n# Evaluate the model\nmae = model.evaluate(X_test, y_train)\nprint(f"MAE: {mae}")\n\n```\n\n# Execution output\n\n```\nTraceback (most recent call last):\n  File "runfile.py", line 31, in <module>\n    X_test = test_df.drop(["target"], axis=1)\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File "/opt/conda/envs/agent/lib/python3.11/site-packages/pandas/core/frame.py", line 5344, in drop\n    return super().drop(\n           ^^^^^^^^^^^^^\n  File "/opt/conda/envs/agent/lib/python3.11/site-packages/pandas/core/generic.py", line 4711, in drop\n    obj = obj._drop_axis(labels, axis, level=level, errors=errors)\n          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File "/opt/conda/envs/agent/lib/python3.11/site-packages/pandas/core/generic.py", line 4753, in _drop_axis\n    new_axis = axis.drop(labels, errors=errors)\n               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File "/opt/conda/envs/agent/lib/python3.11/site-packages/pandas/core/indexes/base.py", line 7000, in drop\n    raise KeyError(f"{labels[mask].tolist()} not found in axis")\nKeyError: "[\'target\'] not found in axis"\nExecution time: 9 seconds seconds (time limit is 9 hours).\n```\n'}],
    tools=tools,
    tool_choice=tool_choice
)

print(completion)

assert 1==0









"""Backend for OpenAI API."""
import json
import logging
import time
import os

from aide.backend.utils import (
    FunctionSpec,
    OutputType,
    opt_messages_to_list,
    backoff_create,
)
from funcy import notnone, once, select_values
import openai

logger = logging.getLogger("aide")

_client: openai.OpenAI = None  # type: ignore


OPENAI_TIMEOUT_EXCEPTIONS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)


@once
def _setup_openai_client():
    global _client
    _client = openai.OpenAI(max_retries=0)

def _setup_llama_client():
    global _client
    _client = openai.OpenAI(
        api_key=os.environ["LLAMA_API_KEY"],
        base_url="https://api.llama.com/compat/v1/",
        max_retries=0
    )

def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    convert_system_to_user: bool = False,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    
    if model_kwargs.get("model", "").startswith("Llama-"):
        _setup_llama_client()
    else:
        _setup_openai_client()
    filtered_kwargs: dict = select_values(notnone, model_kwargs)  # type: ignore

    messages = opt_messages_to_list(
        system_message, user_message, convert_system_to_user=convert_system_to_user
    )

    if func_spec is not None:
        filtered_kwargs["tools"] = [func_spec.as_openai_tool_dict]
        # force the model the use the function
        filtered_kwargs["tool_choice"] = func_spec.openai_tool_choice_dict

    t0 = time.time()
    completion = backoff_create(
        _client.chat.completions.create,
        OPENAI_TIMEOUT_EXCEPTIONS,
        messages=messages,
        **filtered_kwargs,
    )
    req_time = time.time() - t0

    choice = completion.choices[0]

    if func_spec is None:
        output = choice.message.content
    else:
        print('choice.message', choice.message)
        print('messages', messages)
        print('tools', filtered_kwargs["tools"])
        assert (
            choice.message.tool_calls
        ), f"function_call is empty, it is not a function call: {choice.message}"
        assert (
            choice.message.tool_calls[0].function.name == func_spec.name
        ), "Function name mismatch"
        try:
            output = json.loads(choice.message.tool_calls[0].function.arguments)
        except json.JSONDecodeError as e:
            logger.error(
                f"Error decoding the function arguments: {choice.message.tool_calls[0].function.arguments}"
            )
            raise e

    in_tokens = completion.usage.prompt_tokens
    out_tokens = completion.usage.completion_tokens

    info = {
        "system_fingerprint": completion.system_fingerprint,
        "model": completion.model,
        "created": completion.created,
    }

    return output, req_time, in_tokens, out_tokens, info
