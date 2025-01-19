# How the Anote Codebase Works

## Product Structure

The Anote Codebase has a ```frontend``` with our React and Node.js based frontend code, and a ```server``` with our python and flask based backend code. We use a ```mysql``` database defined in ```schema.sql```, that can be initialized via the ```init_db_dev.py``` file. We use AWS for storage, EC2 instance and model GPUs, Docker for Deployment, Kubernetes for container orchestration, and Ray for the parallelization of model specific tasks.

### Frontend

```plaintext
frontend/
├── node_modules/                 # Dependency modules installed via npm
├── public/                       # Static public assets (HTML, images, etc.)
├── src/                          # Source folder for main application code
│   ├── components/               # Reusable React components
│   │   ├── Account.js            # Account management component
│   │   ├── Annotation.js         # Handles data annotation tasks
│   │   ├── Billing.js            # Billing and payment management
│   │   ├── Customize.js          # Customize screen on label component
│   │   ├── Download.js           # Download screen on label component
│   │   ├── DownloadPrivateGPT.js # Download private version of chatbot on Mac, Windows, Linux
│   │   ├── Evaluate/             # Evaluation flow -related components
│   │   ├── GetStarted.js         # Onboarding or introduction page
│   │   ├── Header.js             # Main header component
│   │   ├── MainNavbar.js         # Primary navigation bar
│   │   ├── Navbar.js             # Secondary navigation bar
│   │   ├── Predict/              # Predict flow - related components
│   │   ├── Reviewer.js           # Labeler review-mode components
│   │   ├── Structured.js         # Uploading Structured files
│   │   ├── StructuredDashboard.js # Evaluation Dashboard for structured data
│   │   ├── StructuredMislabels.js # Evaluation Dashboard for mislabeled data
│   │   ├── Train/                # Train flow - related components
│   │   ├── Upload.js             # File upload component
│   │   ├── Version.js            # Model versioning / my Models table
│   │
│   ├── constants/                # Constants used across the app
│   │   ├── DarkColorConstants.js # Dark mode color configurations
│   │   ├── DbEnums.js            # Database enumerations
│   │   ├── RouteConstants.js     # Defined routes for navigation
│   │
│   ├── docs/                     # Documentation files (excluded)
│   ├── homepage/                 # Homepage-specific components and assets
│   ├── http/                     # HTTP client and API utilities
│   │   ├── RequestConfig.js      # Configuration for HTTP requests
│   │
│   ├── landing_page/             # Landing page design (largely excluded)
│   │   ├── Home.js               # Landing page logic
│   │   ├── landing_page_assets/  # Assets for landing page
│   │   ├── LandingPage.js        # Main landing page component
│   │
│   ├── metrics_components/       # Metrics and analytics components (excluded)
│   │   ├── Metrics.js            # General metrics handling
│   │   ├── MetricsDashboard.js   # Dashboard for metrics visualization
│   │   ├── MetricsRoutes.js      # Routes for metrics-related views
│   │
│   ├── redux/                    # Redux state management
│   │   ├── DatasetMetricsSlice.js
│   │   ├── DatasetSlice.js
│   │   ├── DocumentSlice.js
│   │   ├── Global.js
│   │   ├── LabelingFunctionSlice.js
│   │   ├── MetricsSlice.js
│   │   ├── ParsedTextBlockSlice.js
│   │   ├── UserSlice.js
│   │
│   ├── stores/                   # State management
│   │   ├── schema.js
│   │   ├── store.js
│   │
│   ├── styles/                   # Global CSS styling
│   ├── subcomponents/            # Smaller reusable components
│   │   ├── account/
│   │   ├── annotate/
│   │   ├── api/
│   │   ├── commonflows/
│   │   ├── customize/
│   │   ├── dropdown/
│   │   ├── getWindowDimensions.js
│   │   ├── login/
│   │   ├── modals/
│   │   ├── models/
│   │   ├── payments/
│   │   ├── reviewer/
│   │   ├── structured/
│   │   ├── tables/
│   │   ├── upload/
│   │
│   ├── util/                     # Utility functions
│   │   ├── CustomizeUtil.js
│   │   ├── DocumentUtil.js
│   │   ├── DomainParsing.js
│   │   ├── RobotHeader.js
│   │   ├── TableUtil.js
│   │   ├── TaskUtil.js
│   │   ├── TreeUtil.js
│   │
│   ├── zustand/                  # Zustand state for tailwind
│   │   ├── zustand.js
│   │
│   ├── App.js                    # Main React application entry point
│   ├── App.test.js               # Tests for the App component
│   ├── Dashboard.js              # Main dashboard logic
│   ├── index.css                 # Global CSS
│   ├── index.js                  # React DOM entry point
│   ├── permissions.json          # JSON for permissions handling
│   ├── reportWebVitals.js        # Web performance metrics
│   ├── setupTests.js             # Testing setup
│
├── .env.development              # Development environment variables
├── .env.production               # Production environment variables
├── .gitignore                    # Files to exclude from Github
```

### Frontend Subcomponents

There are a lot of details for each of the subcomponents, as seen below:

```plaintext
subcomponents/
├── account/                     # Components related to account management
│   ├── AdminView.js             # Admin-specific account view
│   ├── BillingView.js           # Billing and payment view
│   ├── UserView.js              # General user account view
│
├── annotate/                    # Annotation-related components
│   ├── DefaultAnnotateView.js   # Default annotation interface
│   ├── FileChooser.js           # Component for selecting files for annotation
│
├── api/                         # Components interacting with APIs
│   ├── APIKeyDashboard.js       # Manage API keys in a dashboard view
│
├── commonflows/                 # Common flow-related utilities
│   ├── FlowCsvNextModal.js      # Modal for the next step in CSV flows
│   ├── FlowCsvSelector.js       # Component to select CSV files for flow
│   ├── FlowCsvSelectorTrain.js  # CSV selector specifically for training
│   ├── FlowCsvViewer.js         # Viewer for CSV data in flows
│   ├── FlowDataSetter.js        # Sets up flow-related data
│   ├── UploadDocumentBank.js    # Upload bank of documents for flow
│
├── customize/                   # Components related to UI customization
│   ├── CategoriesTree.js        # Component for a tree-based category structure
│
├── dropdown/                    # Dropdown components
│   ├── DropdownAddColumn.js     # Add column dropdown functionality
│   ├── DropdownPredict.js       # Dropdown for prediction-related actions
│
├── getWindowDimensions.js       # Utility to get window dimensions for responsive design
│
├── login/                       # Components for login and authentication
│   ├── ForgotPasswordComponent.js # Forgot password functionality
│   ├── LoginComponent.js        # User login interface
│   ├── PasswordReset.js         # Password reset flow
│   ├── SignUpComponent.js       # User signup interface
│
├── modals/                      # Reusable modal components
│   ├── AddTrainingModels.js     # Modal for adding training models
│   ├── DeleteDialog.js          # Confirmation dialog for deletions
│
├── models/                      # Model-related components
│   ├── ModelKeysDashboard.js    # Dashboard for managing model keys
│
├── payments/                    # Payments-related components
│   ├── PaymentsComponent.js     # Main payment component
│   ├── PaymentsProduct.js       # Payment product management
│   ├── PricingProduct.js        # Pricing information component
│
├── reviewer/                    # Review-related components
│   ├── AccessControl.js         # Access control for review components
│   ├── AnnotatorMetrics.js      # Metrics for annotators
│   ├── Override.js              # Override-related review functionality
│
├── structured/                  # Components for structured data handling
│   ├── SelectedMultiColumnList.js # Multi-column list for structured data
│   ├── StructuredUploadTree.js  # Tree view for structured data uploads
│
├── tables/                      # Components for handling table-related data
│   ├── annotation_history/      # History of annotations
│   ├── download/                # Table for downloads
│   ├── evaluate_flow/           # Evaluation-related flow tables
│   ├── labeling_fn/             # Tables for labeling functions
│   ├── Pagination.js            # Pagination component for tables
│   ├── predict_flow/            # Tables for prediction flow data
│   ├── review/                  # Tables for review processes
│
├── upload/                      # Upload-related components
│   ├── create_dataset_assets/   # Components for creating dataset assets
│   ├── CreateDatasetView.js     # Interface for creating datasets
│   ├── dataChooserViews/        # Data chooser views for upload flows
│   ├── DatasetInfoModal.js      # Modal for dataset information
│   ├── GetStartedView.js        # View for starting uploads
│   ├── NewDatasetView.js        # View for creating new datasets
│   ├── SelectDatasetView.js     # View for selecting existing datasets
│   ├── UploadNoUserSession.js   # Upload functionality without user sessions
│   ├── UploadSelectedModal.js   # Modal for selected upload options
│   ├── UploadStepper.js         # Stepper component for upload flows
│   ├── UploadUnselected.js      # Component for unselected uploads
```

### Backend

``` plaintext
backend/
├── api_endpoints/                # API endpoint definitions for core functionality
│   ├── add_annotator/            # Add a single annotator to a project
│   ├── ...                       # (Many other endpoint modules for CRUD and workflows)
│   └── view_user/                # View user-related details
│
├── app.py                        # Main application entry point
│
├── aws_deploy/                   # AWS deployment scripts and configurations
│
├── client_secret.json            # Client secret for API or service authentication (secure this!)
│
├── connectors/                   # Modules to connect with external services/APIs
│   ├── asana.py                  # Integration with Asana API
│   ├── github.py                 # Integration with GitHub API
│   ├── huggingface.py            # Integration with Hugging Face API
│   ├── notion.py                 # Integration with Notion API
│   ├── reddit.py                 # Integration with Reddit API
│   ├── s3.py                     # AWS S3 storage operations
│   ├── salesforce.py             # Integration with Salesforce API
│   ├── shopify.py                # Integration with Shopify API
│   ├── slack.py                  # Slack API integration for notifications
│   ├── snowflake.py              # Integration with Snowflake database
│   └── twitter.py                # Integration with Twitter API
│
├── constants/                    # Application-wide constants
│   ├── global_constants.py       # Constants shared across modules
│   └── create_eks_cluster.yaml   # YAML file for creating an AWS EKS cluster
│
├── database/                     # Database-specific files and configurations
│   ├── db_auth.py                # Authentication logic for database access
│   ├── db.py                     # Core database operations and utilities
│   ├── init_db_dev.py            # Initialization script for development database
│   ├── init_db_prod.py           # Initialization script for production database
│   ├── schema.sql                # SQL schema for database tables
│   ├── schema.txt                # Text-based description of database schema
│   └── db_enums.py               # Enumerations for database fields
│
├── Dockerfile                    # Base Docker configuration
├── DockerfileRayCpu              # Dockerfile optimized for Ray on CPUs
├── DockerfileRayGpu              # Dockerfile optimized for Ray on GPUs
│
├── engineering/                  # Engineering-specific tools and scripts
│   ├── decomposer.py             # Script for decomposing engineering tasks
│
├── gtm_scripts/                  # Go-to-market Chatbot Scripts
│
├── kubernetes_ray.yaml           # Kubernetes configuration for Ray
├── kubernetes.yaml               # General Kubernetes deployment configuration
│
├── models/                       # Machine Learning models and utilities
│   ├── labeling_function/        # Modules for labeling functions
│   │   ├── entities/             # Entity-related labeling
│   │   ├── labeling_functions.py # Custom labeling logic
│   │   └── regex/                # Regex-based labeling methods
│   ├── metrics/                  # Metrics computation modules
│   │   ├── ClassificationMetrics.py # Metrics for classification tasks
│   │   ├── NERMetrics.py         # Named Entity Recognition metrics
│   │   └── PromptingMetrics.py   # Metrics for prompting performance
│   ├── model_training.py         # Main training pipeline for models
│   ├── model_util.py             # Utilities for model handling
│   ├── run_async_models.py       # Asynchronous model execution script
│   └── zero_shot_models.py       # Zero-shot model configuration
│
├── ray_tasks.py                  # Distributed computing tasks using Ray
├── requirements.txt              # Python package dependencies
│
├── sdk/                          # SDK-related files
│   ├── anoteai/                  # Core SDK modules within core.py
│   ├── examples/                 # Example SDK usage scripts
│   ├── setup.py                  # Setup script for packaging the SDK
│   └── anoteai.egg-info/         # Metadata for the Python package
│
├── stripe_config/                # Stripe payment integration configuration
│
├── util/                         # Utility scripts for backend operations
│   ├── call_llm.py               # Call LLM APIs and handle responses
│   ├── package_size.py           # Analyze and handle package sizes
│   └── timer.py                  # Timer utilities for measuring performance
│
└── wsgi.py                       # WSGI configuration for hosting the Python application
```

#### api endpoints
When an API handler is defined in ```app.py```, there is an associated api endpoint defined in the API Endpoints folder.

``` plaintext
api_endpoints/
├── add_annotator/               # Add a single annotator to a project or dataset
├── add_annotators/              # Add multiple annotators in bulk
├── add_datasets_to_project/     # Link datasets to specific projects
├── build_your_own_gpt/          # Build custom GPT models
├── chatbot_category/            # Manage chatbot categories
├── companies/                   # Manage company-related endpoints
├── create_category/             # Create new categories
├── create_dataset/              # Create a new dataset
├── create_document_url/         # Generate URLs for document access
├── create_labeling_function/    # Create a new labeling function
├── create_project/              # Create a new project
├── create_visit/                # Log user visits or activities
├── delete_annotations/          # Delete specific annotations
├── delete_api_key/              # Delete API keys
├── delete_category/             # Delete categories
├── delete_dataset/              # Remove datasets
├── delete_labeling_function/    # Delete a specific labeling function
├── download_labeled_data/       # Download labeled dataset
├── download_tagged_data/        # Download tagged data for a dataset
├── financeGPT/                  # Specialized endpoints for Finance GPT-related tasks
├── generate_api_key/            # Generate new API keys
├── get_api_keys/                # Retrieve existing API keys
├── initialize_model/            # Initialize and configure models
├── label_data/                  # Label data points in datasets
├── login_with_email/            # Email login for user authentication
├── models/                      # Manage and retrieve models
├── override_annotations/        # Override existing annotations
├── payments/                    # Payment-related endpoints
├── predict/                     # Handle prediction-related tasks
├── predict_report/              # Generate prediction reports
├── public_apis/                 # Publicly available API endpoints
├── public_evaluation/           # Public-facing evaluation endpoints
├── public_evaluation_rag/       # Evaluation for Retrieval-Augmented Generation (RAG)
├── recompose/                   # Recompose data or workflows
├── remove_from_project/         # Remove items or users from projects
├── save_advanced_settings/      # Save advanced user settings
├── sign_up/                     # User sign-up endpoint
├── skip_annotation/             # Skip specific annotations
├── soft_delete/                 # Soft-delete records
├── upload_connector_data/       # Upload data for connector integrations
├── upload_files/                # Handle file uploads
├── upload_html_data/            # Upload HTML-specific data
├── versions/                    # Manage and retrieve version details
├── view_admin_access_control_df/ # View admin access control data
├── view_annotator_metrics/      # View metrics for annotators
├── view_annotators/             # View available annotators
├── view_categories/             # View system categories
├── view_dataset_metrics/        # View metrics for datasets
├── view_dataset_review_df/      # Review data for datasets
├── view_datasets/               # View all datasets
├── view_documents/              # View documents
├── view_entities/               # View extracted entities
├── view_k_last_annotations/     # View the last K annotations
├── view_k_next_parsed_text_blocks/ # View next parsed text blocks
├── view_labeling_fn_columns_df/ # View columns for labeling functions
├── view_labeling_fn_coverage_df/# View coverage metrics for labeling functions
├── view_labeling_functions/     # View available labeling functions
├── view_metrics/                # View overall system metrics
├── view_mislabels_df/           # View mislabeled items
├── view_model/                  # View details of specific models
├── view_parsed_text_block_df/   # View parsed text blocks in a dataset
├── view_projects/               # View all projects
├── view_structured_accuracy_df/ # View structured accuracy data
├── view_user/                   # View user-related information
```

We leverage ```flask APIs``` in our backend.

### Database

The ```db.py``` file contains all of the SQL operations done by each API endpoint. For instance, when we call an ```add_annotator``` endpoint, we will have the SQL statement in this file to add a user to the user table defined in ```schema.sql``` in this ```db.py``` file. ```schema.sql``` - contains our database schema, which has the following tables:
![https://github.com/nv78/Anote/blob/main/materials/images/Database.png?raw=true](https://github.com/nv78/Anote/blob/main/materials/images/codebase/Database.png)

## Example API Walkthrough - Post-Training Models

On our Customize Screen in our frontend, specifically in ```subcomponents/customize/CategoriesTree.js``` of our Label Flow, we have a Button called **Train Model** that looks like the image below.

![alt text](https://github.com/nv78/OpenAnote/blob/main/materials/images/codebase/DoTrain.png?raw=true)

``` javascript
<Button
  color="success"
  className={"w-full my-3.5"}
  onClick={() => {
    alert('The large language model is now training. In the background, models like GPT-4, Claude, or Llama2 are compiling initial predictions / answers to your prompts. These predictions should render on the annotate and download pages shortly.');
    dispatch(initializeModel({"id": currentDataset}));
  }}
>
  Train Model
</Button>
```

On press of this button, we would like to train a model. To do this, we dispatch a model training job for the current dataset ID. This calls an async thunk within our ```redux/DatasetSlice.js``` file, which await the results of the API call from the initializeModel endpoint in our backend.
``` javascript
export const initializeModel = createAsyncThunk("dataset/initializeModel", async (payload, thunk) => {
    const response = await fetcher("initializeModel", {
        method: "POST",
        headers: {
        'Accept': 'application/json',
        'Content-type': 'application/json',
        },
        body: JSON.stringify(payload)
    })
    const response_str = await response.json();
    return { response_str };
});
```

This backend handler is in ```app.py```. Once we verify that the user is authenticated / logged in, and has access to the relevant resources, we can call our API endpoint.

``` python
@app.route('/initializeModel', methods = ['POST'])
@jwt_or_session_token_required
def InitializeModel():
  verifyAuthForIDs(ProtectedDatabaseTable.DATASETS, request.json["id"])
  predict_report_id = request.json.get("predictReportId")
  if predict_report_id:
    verifyAuthForIDs(ProtectedDatabaseTable.PREDICT_REPORTS, predict_report_id)
  return jsonify(InitializeModelHandler(request))
```
Our API endpoint can be found within ```backend/api_endpoints/initialize_model/handler.py```. 

``` python
def InitializeModelHandler(request):
    dataset_id = request.json["id"]
    taskType = task_for_dataset(dataset_id)
    modelTypes = None
    ...

    modelObjs = []
    roots = view_multi_column_roots(dataset_id)

    for root in roots:

        models = create_models_for_root(root["id"], taskType, overriddenModelTypes=modelTypes)

        for model in models:
            model_id = model["id"]
            modelObjs.append({
                "id": model_id,
                "multiColumnRootsId": root["id"],
                "modelType": model["model_type"],
            })
            train_and_label_all_single_root_single_model.remote(dataset_id, root["id"], model, None)

    return {
        'status': 'OK',
        'datasetId': dataset_id,
        'models': modelObjs
    }
```

Since we don't have a predict report, for each model we want to train, for each multi column root that we want to train on, we will call ```train_and_label_all_single_root_single_model```.

``` py

@ray.remote
def train_and_label_all_single_root_single_model(datasetId, multiColumnRootsId, model, documentBankId):
    modelType = ModelType(model["model_type"])
    if not documentBankId:
        # Train model
        ray.get(train_model_helper.remote(
            datasetId,
            multiColumnRootsId,
            model,
            True,
            None,
            None,
            None,
            None
        ))
    ...

    if not documentBankId:
        merge_predictions(datasetId, multiColumnRootsId, modelType)
```

If the ModelType is supported within our current modelTypes enum, we can begin the training call.

``` python
class ModelType(IntEnum):
    NO_LABEL_TEXT_CLASSIFICATION = 0
    FEW_SHOT_TEXT_CLASSIFICATION = 1
    NAIVE_BAYES_TEXT_CLASSIFICATION = 2
    SETFIT_TEXT_CLASSIFICATION = 3
    NOT_ALL_TEXT_CLASSIFICATION = 4
    FEW_SHOT_NAMED_ENTITY_RECOGNITION = 5
    EXAMPLE_BASED_NAMED_ENTITY_RECOGNITION = 6
    GPT_FOR_PROMPTING = 7
    PROMPT_NAMED_ENTITY_RECOGNITION = 8
    PROMPTING_WITH_FEEDBACK_PROMPT_ENGINEERED = 9
    DUMMY = 10
    GPT_FINETUNING = 11
    RAG_UNSUPERVISED = 12
    ZEROSHOT_GPT4 = 13
    ZEROSHOT_CLAUDE = 14
    ZEROSHOT_LLAMA3 = 15
    ZEROSHOT_MISTRAL = 16
    ZEROSHOT_GPT4MINI = 17
    ZEROSHOT_GEMINI = 18
```

We can train that model, or use that model to make labels using the ```train_model_helper``` module. Our model logic incorporates ray for parallelization. Depending on the task type, we choose a specific model to call.

``` python
import ray
from models.model_training import train_model_helper, merge_predictions
from database.db import update_accuracy

def ray_remote_decorator():
    not_is_prod = os.environ.get("IS_PROD", "false")
    if not not_is_prod:
        return ray.remote(num_gpus=0.2, num_cpus=1)
    else:
        return ray.remote(num_cpus=1)

@ray_remote_decorator()
def run_few_shot_model(datasetId, oldCategories, newCategories):
    print("few shot model")
    rootIds = view_multi_column_roots(datasetId)

    for root in rootIds:
        rootId = root["id"]

        modelType = ModelType.FEW_SHOT_TEXT_CLASSIFICATION
        # Train model
        train_model_helper(datasetId, rootId, modelType)
        # Update database with predictions and probabilitys
        merge_predictions(datasetId, rootId, modelType)
        # Update accuracy table
        update_accuracy(datasetId, rootId, oldCategories, newCategories)
```

The ```train_model_helper.py``` routes to most of our ```model_training``` logic, and is where some of our models are instantiated. It also contains ```merge_predictions``` which returns and route the best LLMs to the end user, based on each of the training jobs.

``` python
def merge_predictions(datasetId, rootId, modelType):
    tracker = TimingTracker(datasetId, "merge_predictions: " + modelType.name)
    tracker.start()
    taskType = task_for_dataset(datasetId)
    isStructuredPrompt = is_structured_prompt(taskType, rootId)
    preds = preds_for_dataset(datasetId, rootId, taskType, isStructuredPrompt)
    mergedPredictions = [merge_prediction(pred, taskType, isStructuredPrompt) for pred in preds]
    update_predicted_and_probabilities(mergedPredictions, taskType, rootId, isStructuredPrompt)
    tracker.end()
```

Our merge predictions enables you to ensemble each model and choose the best LLM. There are different ways to ensemble and merge each model, depending on what works best for your use case. Once completed, we want to update our database fields to store this information in a way that we can return to the end user.

``` python
def update_predicted_and_probabilities(mergedPredictions, taskType, multiColumnRootsId, isStructuredPrompt):
    conn, cursor = get_db_connection()
    ...
    elif taskType == NLPTask.PROMPTING or taskType == NLPTask.CHATBOT:
        for row in mergedPredictions:
            id = row["id"]
            preds = row["preds"]

            cursor.execute('SELECT predicted_multi_columns_id FROM parsedTextBlocks WHERE is_labelable = 1 AND id = %s', [id])
            predictedIdDb = cursor.fetchone()
            if predictedIdDb and predictedIdDb["predicted_multi_columns_id"]:
                predictedId = predictedIdDb["predicted_multi_columns_id"]
            else:
                cursor.execute('INSERT INTO multiColumns () VALUES ()')
                cursor.execute('SELECT LAST_INSERT_ID()')
                predictedId = cursor.fetchone()["LAST_INSERT_ID()"]
                cursor.execute('UPDATE parsedTextBlocks SET predicted_multi_columns_id = %s WHERE is_labelable = 1 AND id = %s', [predictedId, int(id)])

            cursor.execute('DELETE c FROM multiColumns p JOIN multiColumn c ON c.multi_columns_id=p.id WHERE multi_column_roots_id=%s AND p.id = %s', [multiColumnRootsId, predictedId])

            # Add prompt outputs to multiColumn table
            for pred in preds:

                cursor.execute('INSERT INTO multiColumn (multi_columns_id, multi_column_roots_id, category_id, prompt_output, citation_id) VALUES (%s, %s, %s, %s, %s)', [predictedId, multiColumnRootsId, pred["category_id"], pred["prompt_output"], pred["chunk_id"]])

    conn.commit()
    conn.close()
```

This is essentially doing SQL statements to update the tables mentioned in schema.sql for the given user. Once the initialize_model API endpoint is completed, we are able to have the python fields that we return to the user via the API endpoint

``` python
return {
    'status': 'OK',
    'datasetId': dataset_id,
    'models': modelObjs
}
```
Then via the handler in app.py in JSON mode.

``` python
return jsonify(InitializeModelHandler(request))
```

And then back within the redux Dataset Slice
``` js
const response_str = await response.json();
    return { response_str };
```

All the way back to the user on the frontend. When the model training is done, the ```My Models``` table should be updated with the associated modelID for that latest model training job.

![models](https://github.com/nv78/OpenAnote/blob/main/materials/images/models.png?raw=true)

You can call that model ID within our SDK to make predictions via API by following [the instructions here](https://docs.anote.ai/api-anote/setup.html).

### Deployments

To successfully deploy this code to [production](https://dashboard.anote.ai/), you will first need to have [docker desktop](https://www.docker.com/products/docker-desktop/) installed. You will also need our AWS Credentials, or you can create your own credentials depending on where you would like to deploy code modifications. We leverage a variety of different AWS services, including ```AWS S3``` for frontend storage, ```AWS Cloudfront``` for invalidating distributions, ```AWS Route53``` for domain linkage, and ```AWS ECR``` for EC2 instances with GPUs, amongst other services that you might need to pre-configure. From there, you can use the following commands:

#### Frontend Deployment Command
``` python
$ REACT_APP_API_ENDPOINT=https://api.anote.ai npm run build && for file in ./build/static/js/*.js; do uglifyjs "$file" --compress --mangle -o "$file"; done && aws s3 sync build/ s3://anote-product-frontend --acl public-read
```

#### Documentation Deployment Command
``` python
$ mkdocs build
$ aws s3 sync site/ s3://anote-product-docs --acl public-read
```

#### Backend Deployment Command
``` python
$ docker build -t anote-backend . --platform linux/amd64
$ docker run -p 5000:5000 anote-backend (To test that it works.  With the server running, you can manually test on the react frontend)
$ docker tag anote-backend:latest ACCOUNT_NUMBER.dkr.ecr.us-east-1.amazonaws.com/anote-backend:latest
$ docker push ACCOUNT_NUMBER.dkr.ecr.us-east-1.amazonaws.com/anote-backend:latest
```

If you’re not logged into AWS, the push may fail and you would have to run:

``` python
$ aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin ACCOUNT_NUMBER.dkr.ecr.us-east-1.amazonaws.com
```

Now your docker build with all the new server code will be saved as an “image” to ECR.  We want to take an instance of this image (a container) and deploy it to Elastic Beanstalk.  We need to

``` python
$ cd server/aws_deploy
```

Because this contains the Dockerrun.aws.json file that references our ECR docker image.  Then we can run:

```
$ aws credentials
$ eb init (and select Anote2)
$ eb deploy
```

#### PyPi Package Deployment for SDK

Once you modify the ```API_BASE_URL``` from `http://localhost:5000` to `https://api.anote.ai` in `server/sdk/anoteai/core.py`, you will need to increment the version number in `server/sdk/setup.py`. From there, you can run the following commands to deploy the updated version of the SDK:

``` python
$ python setup.py sdist bdist_wheel
$ twine upload dist/*
```

You will need access to our pypi package token. To test this in prod, you can install the upgraded version of the pypi package:

```
$ pip install -U anoteai
```

Then you can create an API key on https://dashboard.anote.ai/, and test out the updated pypi package calls.
