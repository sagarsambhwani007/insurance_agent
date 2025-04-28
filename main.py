# --- Imports ---
import os
import re
import json
import uuid
import openai
import numpy as np
import fitz  # PyMuPDF
from datetime import datetime
from typing import List, Dict, Any, Optional
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.memory import ConversationBufferWindowMemory
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# --- Set your OpenAI Key ---
openai.api_key = os.getenv("OPENAI_API_KEY")  # or paste key directly

# --- Initialize Embedding Model and LLM ---
embedding_model = OpenAIEmbeddings()
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)
fact_check_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)  # For fact checking

# --- Initialize Memory with window size of 5 ---
conversation_memory = ConversationBufferWindowMemory(k=5)

# --- Load or Create Policy Database ---
PERSIST_DIR = "./icici_policy_chroma_db"
SESSION_DIR = "./user_sessions"
POLICY_LIST_FILE = "./policy_list.json"

# Create necessary directories
os.makedirs(SESSION_DIR, exist_ok=True)

# --- Define Pydantic Models for Structured Data ---
class PolicyEntity(BaseModel):
    policy_name: str = Field(description="The name of the policy")
    policy_type: Optional[str] = Field(None, description="Type of policy (health, life, etc.)")
    mentioned_in_turn: int = Field(description="Conversation turn where this policy was mentioned")
    source_document: Optional[str] = Field(None, description="Source document for this policy")

class UserPreference(BaseModel):
    preference_type: str = Field(description="Type of preference (age, family, budget, etc.)")
    value: str = Field(description="The value of this preference")
    confidence: float = Field(description="Confidence score (0-1) of this preference")
    
class UserProfile(BaseModel):
    user_id: str = Field(description="Unique ID for the user")
    preferences: List[UserPreference] = Field(default_factory=list, description="List of user preferences")
    policies_discussed: List[PolicyEntity] = Field(default_factory=list, description="Policies discussed with this user")
    policies_recommended: List[PolicyEntity] = Field(default_factory=list, description="Policies recommended to this user")
    last_interaction: str = Field(description="Timestamp of last interaction")

# --- Load or Create Policy Database ---
if not os.path.exists(PERSIST_DIR):
    print("\n:warning: No ChromaDB found. Loading PDFs and creating DB...")
    
    def load_pdf_text(pdf_folder):
        documents = []
        policy_list = []
        for filename in os.listdir(pdf_folder):
            if filename.endswith(".pdf"):
                filepath = os.path.join(pdf_folder, filename)
                pdf_doc = fitz.open(filepath)
                full_text = ""
                for page in pdf_doc:
                    full_text += page.get_text()
                policy_name = os.path.splitext(filename)[0].replace("_", " ").replace("-", " ").strip()
                documents.append({"text": full_text, "source": filename, "policy_name": policy_name})
                
                # Extract basic policy info for our policy list
                policy_info = {
                    "policy_name": policy_name,
                    "source": filename,
                    "policy_type": "unknown"  # We'll try to extract this later
                }
                policy_list.append(policy_info)
                pdf_doc.close()
        
        # Save policy list to file for reference
        with open(POLICY_LIST_FILE, 'w') as f:
            json.dump(policy_list, f, indent=2)
            
        return documents

    def chunk_text(documents, chunk_size=500, chunk_overlap=50):
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        doc_objects = []
        for doc in documents:
            chunks = splitter.split_text(doc["text"])
            for chunk in chunks:
                doc_objects.append(Document(
                    page_content=chunk,
                    metadata={"source": doc["source"], "policy_name": doc["policy_name"]}
                ))
        return doc_objects
    
    pdf_folder = "./icici_policies"  # <--- Folder where your ICICI PDFs are kept
    documents = load_pdf_text(pdf_folder)
    docs = chunk_text(documents)
    chroma_db = Chroma.from_documents(docs, embedding_model, persist_directory=PERSIST_DIR)
    chroma_db.persist()
    print("\n:white_check_mark: Policy ChromaDB created successfully!")
else:
    chroma_db = Chroma(persist_directory=PERSIST_DIR, embedding_function=embedding_model)
    print("\n:white_check_mark: Loaded existing Policy ChromaDB.")
    
    # Load policy list if it exists, or create an empty one
    if os.path.exists(POLICY_LIST_FILE):
        with open(POLICY_LIST_FILE, 'r') as f:
            policy_list = json.load(f)
    else:
        # Create policy list from metadata in the database
        policy_list = []
        collection = chroma_db._collection
        metadatas = collection.get()["metadatas"]
        seen_policies = set()
        
        for metadata in metadatas:
            if metadata and "policy_name" in metadata:
                if metadata["policy_name"] not in seen_policies:
                    policy_info = {
                        "policy_name": metadata["policy_name"],
                        "source": metadata.get("source", "unknown"),
                        "policy_type": "unknown"
                    }
                    policy_list.append(policy_info)
                    seen_policies.add(metadata["policy_name"])
        
        with open(POLICY_LIST_FILE, 'w') as f:
            json.dump(policy_list, f, indent=2)

# --- Session Management ---
def get_or_create_session(user_id: str) -> UserProfile:
    """Get an existing user session or create a new one"""
    session_file = os.path.join(SESSION_DIR, f"{user_id}.json")
    if os.path.exists(session_file):
        with open(session_file, 'r') as f:
            data = json.load(f)
            return UserProfile(**data)
    else:
        # Create new profile
        profile = UserProfile(
            user_id=user_id,
            preferences=[],
            policies_discussed=[],
            policies_recommended=[],
            last_interaction=datetime.now().isoformat()
        )
        save_session(profile)
        return profile
def save_session(profile: UserProfile):
    """Save user session to file"""
    session_file = os.path.join(SESSION_DIR, f"{profile.user_id}.json")
    # Fix for Pydantic v2 deprecation warning - using model_dump instead of dict
    with open(session_file, 'w') as f:
        json.dump(profile.model_dump(), f, indent=2)
# --- Advanced Entity Extraction System ---
def extract_policy_entities(text: str, conversation_turn: int) -> List[PolicyEntity]:
    """Extract policy names using LLM-based entity extraction"""
    # First, check against known policy list for exact matches
    extracted_policies = []
    # Load policy list from file for reference
    try:
        with open(POLICY_LIST_FILE, 'r') as f:
            policy_list = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        # If policy list file doesn't exist or is corrupted, create an empty list
        policy_list = []
    # Use regex for basic entity extraction
    # Look for patterns like "X Policy", "Y Plan", "Z Insurance", etc.
    policy_patterns = [
        r'([A-Z][a-zA-Z\s]+(?:Policy|Plan|Insurance|Cover|Protection))',
        r'(ICICI\s+[a-zA-Z\s]+)',
        r'([A-Z][a-zA-Z\s]+(?:Term|Health|Life|Motor|Home|Travel)\s+[a-zA-Z\s]+)'
    ]
    found_policies = set()
    for pattern in policy_patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            policy_name = match.group(1).strip()
            if len(policy_name) > 3 and policy_name not in found_policies:  # Avoid very short matches
                found_policies.add(policy_name)
    # Check against our known policy list for exact or fuzzy matches
    for policy_name in found_policies:
        # Check for exact match in policy list
        exact_match = False
        for policy in policy_list:
            if policy_name.lower() == policy["policy_name"].lower():
                extracted_policies.append(
                    PolicyEntity(
                        policy_name=policy["policy_name"],  # Use the canonical name
                        policy_type=policy["policy_type"],
                        mentioned_in_turn=conversation_turn,
                        source_document=policy["source"]
                    )
                )
                exact_match = True
                break
        # If no exact match found, add as a new entity
        if not exact_match:
            extracted_policies.append(
                PolicyEntity(
                    policy_name=policy_name,
                    policy_type=None,  # Unknown type
                    mentioned_in_turn=conversation_turn,
                    source_document=None
                )
            )
    # If pattern matching found nothing, use LLM for extraction
    if not extracted_policies:
        extraction_prompt = """
        Extract any insurance policy names mentioned in the following text. If no specific policy names are mentioned, return an empty list.
        Format the output as a list of policy names. Only include actual policy names, not general descriptions.
        TEXT: {text}
        POLICY NAMES:
        """
        prompt = ChatPromptTemplate.from_template(extraction_prompt)
        chain = prompt | llm | StrOutputParser()
        try:
            result = chain.invoke({"text": text})
            # Process the results - expecting a list of policy names
            if result and result.strip():
                lines = result.strip().split('\n')
                for line in lines:
                    # Clean up the line (remove numbers, dashes, etc.)
                    clean_line = re.sub(r'^[\d\-\.\*\•\s]+', '', line).strip()
                    if clean_line and len(clean_line) > 3:  # Avoid very short matches
                        extracted_policies.append(
                            PolicyEntity(
                                policy_name=clean_line,
                                policy_type=None,  # Unknown type
                                mentioned_in_turn=conversation_turn,
                                source_document=None
                            )
                        )
        except Exception as e:
            print(f"Error in LLM policy extraction: {str(e)}")
    return extracted_policies

# --- Extract User Preferences System ---
def extract_user_preferences(text: str) -> List[UserPreference]:
    """Extract user preferences using LLM"""
    preference_prompt = """
    Extract any user preferences or needs related to insurance from the following text.
    Focus on: age, family details, budget constraints, risk factors, occupation, health conditions, etc.
    
    For each identified preference, provide:
    1. The type of preference (e.g., age, family_size, budget, etc.)
    2. The value of this preference
    3. A confidence score (0-1) of how certain you are about this preference
    
    Format your response as a JSON list with these fields. If no preferences are found, return an empty list.
    
    TEXT: {text}
    
    PREFERENCES:
    """
    
    prompt = ChatPromptTemplate.from_template(preference_prompt)
    chain = prompt | llm | StrOutputParser()
    
    try:
        result = chain.invoke({"text": text})
        
        # Try to parse the result as JSON
        preferences = []
        # Clean up the result to help with JSON parsing
        clean_result = result.strip()
        
        # If result starts with ```json and ends with ```, strip those
        if clean_result.startswith("```json") and clean_result.endswith("```"):
            clean_result = clean_result[7:-3].strip()
        elif clean_result.startswith("```") and clean_result.endswith("```"):
            clean_result = clean_result[3:-3].strip()
            
        # If it's an empty list or specific text indicating empty
        if clean_result == "[]" or "no preferences" in clean_result.lower():
            return []
            
        # Try to parse the JSON
        try:
            parsed = json.loads(clean_result)
            if isinstance(parsed, list):
                for item in parsed:
                    preferences.append(
                        UserPreference(
                            preference_type=item.get("type", "unknown"),
                            value=str(item.get("value", "")),
                            confidence=float(item.get("confidence", 0.5))
                        )
                    )
            return preferences
        except json.JSONDecodeError:
            # If JSON parsing fails, try to extract preferences manually
            # Look for patterns like "type: value (confidence)"
            pattern = r'(\w+):\s*([^(]+)\s*\((\d+\.\d+)\)'
            matches = re.finditer(pattern, clean_result)
            for match in matches:
                pref_type, value, confidence = match.groups()
                preferences.append(
                    UserPreference(
                        preference_type=pref_type.strip(),
                        value=value.strip(),
                        confidence=float(confidence)
                    )
                )
            return preferences
    except Exception as e:
        print(f"Error in preference extraction: {str(e)}")
        return []

# --- Fact Checking Mechanism ---
def fact_check_policy_info(policy_name: str, claimed_info: str) -> bool:
    """Verify if claimed information about a policy is supported by the database"""
    # Query the database for information about this policy
    if not policy_name:
        return False
        
    query = f"Information about {policy_name} policy"
    retriever = chroma_db.as_retriever(search_kwargs={"k": 3})
    docs = retriever.get_relevant_documents(query)
    
    if not docs:
        return False  # No information found to verify against
    
    # Extract context from documents
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Create a fact checking prompt
    fact_check_prompt = """
    You are a precise fact checker for insurance policy information. Verify if the claimed information is supported by the reference text.
    
    POLICY: {policy}
    
    CLAIMED INFORMATION: {claim}
    
    REFERENCE TEXT FROM POLICY DATABASE:
    {context}
    
    Is the claimed information verifiably supported by the reference text? Answer with ONLY "Yes" or "No".
    If there's not enough information to verify, or if the claim contains any details not supported by the reference, answer "No".
    """
    
    prompt = ChatPromptTemplate.from_template(fact_check_prompt)
    chain = prompt | fact_check_llm | StrOutputParser()
    
    try:
        result = chain.invoke({
            "policy": policy_name,
            "claim": claimed_info,
            "context": context
        })
        
        # Parse the result
        result = result.lower().strip()
        if "yes" in result:
            return True
        else:
            return False
    except Exception as e:
        print(f"Error in fact checking: {str(e)}")
        return False

# --- Define Tool Intents with Improved Instructions ---
tool_intents = [
    {
        "tool_name": "recommendation",
        "instruction": "Find the best policy for the user who is asking about insurance recommendations, policy suggestions, or best options based on their profile, needs or requirements."
    },
    {
        "tool_name": "comparison",
        "instruction": "Compare two or more insurance policies when the user wants to know differences, advantages, disadvantages, or evaluate multiple policies against each other."
    },
    {
        "tool_name": "faq",
        "instruction": "Answer questions about policy details, coverage, benefits, premiums, claim processes, or general insurance information that the user is asking about."
    },
    {
        "tool_name": "similar_policy",
        "instruction": "Find policies similar to ones previously mentioned or discussed in the conversation history."
    }
]

# --- Create Embeddings for Tool Intents ---
tool_embeddings = []
tool_names = []
for tool in tool_intents:
    embedding = embedding_model.embed_query(tool["instruction"])
    tool_embeddings.append(embedding)
    tool_names.append(tool["tool_name"])
tool_embeddings = np.array(tool_embeddings)

# --- Define Enhanced Tools with Better Prompting and Memory Integration ---

# Recommendation Tool with LLM and Profile Awareness
def recommend_policy(user_input: str, user_profile: UserProfile, conversation_turn: int):
    """Enhanced recommendation tool that incorporates user profile"""
    # Get conversation history
    conversation_history = conversation_memory.load_memory_variables({})
    history_text = conversation_history.get("history", "")
    # Extract new user preferences from this input
    new_preferences = extract_user_preferences(user_input)
    # Update user profile with new preferences
    if new_preferences:
        # For each new preference, check if we already have it
        for new_pref in new_preferences:
            updated = False
            for i, existing_pref in enumerate(user_profile.preferences):
                if existing_pref.preference_type == new_pref.preference_type:
                    # Update existing preference if new one has higher confidence
                    if new_pref.confidence > existing_pref.confidence:
                        user_profile.preferences[i] = new_pref
                    updated = True
                    break
            if not updated:
                user_profile.preferences.append(new_pref)
    # Format user preferences for the prompt
    preference_text = ""
    if user_profile.preferences:
        preference_text = "User preferences:\n"
        for pref in user_profile.preferences:
            preference_text += f"- {pref.preference_type}: {pref.value}\n"
    # Retrieve relevant documents - Fix for LangChain deprecation warning
    retriever = chroma_db.as_retriever(search_kwargs={"k": 5})
    # Changed from get_relevant_documents to invoke
    docs = retriever.invoke(user_input)
    if not docs:
        return ":x: No matching policy information found in our database."
    # Extract context from documents
    contexts = [doc.page_content for doc in docs]
    sources = [doc.metadata.get("source", "Unknown") for doc in docs]
    # Create a recommendation prompt template with conversation history and user preferences
    recommendation_template = """
    You are a helpful insurance advisor for ICICI. Based on the user's query, conversation history, user preferences, and the available policy information,
    recommend the most suitable insurance policies.
    CONVERSATION HISTORY:
    {history}
    USER QUERY: {query}
    USER PREFERENCES:
    {preferences}
    AVAILABLE POLICY INFORMATION:
    {context}
    Please provide:
    1. A personalized recommendation of the best policy options based on the query and user preferences
    2. Key benefits of each recommended policy that align with user needs
    3. Any important considerations the user should keep in mind
    List each recommended policy with a clear, consistent name that can be referenced later. Bold the policy names for clarity.
    Your response should be clear, concise, and helpful. Format it nicely for readability.
    """
    # Create the prompt
    prompt = ChatPromptTemplate.from_template(recommendation_template)
    # Build the chain
    chain = (
        {"query": RunnablePassthrough(),
         "context": lambda x: "\n\n".join(contexts),
         "history": lambda x: history_text,
         "preferences": lambda x: preference_text}
        | prompt
        | llm
        | StrOutputParser()
    )
    # Run the chain
    response = chain.invoke(user_input)
    # Extract policy entities from the response
    policies = extract_policy_entities(response, conversation_turn)
    # Add these policies to the recommended list in user profile
    for policy in policies:
        # Check if this policy is already in the recommended list
        already_recommended = False
        for existing in user_profile.policies_recommended:
            if existing.policy_name.lower() == policy.policy_name.lower():
                already_recommended = True
                break
        if not already_recommended:
            # Fact check before adding to recommendations
            fact_check_result = True  # Assume true by default
            if policy.policy_name:
                # Extract the part of the response describing this policy
                policy_pattern = rf"(?:.*{re.escape(policy.policy_name)}.*(?:\n.*)*)"
                matches = re.search(policy_pattern, response)
                if matches:
                    policy_description = matches.group(0)
                    fact_check_result = fact_check_policy_info(policy.policy_name, policy_description)
            if fact_check_result:
                user_profile.policies_recommended.append(policy)
                # Also add to discussed
                user_profile.policies_discussed.append(policy)
    # Save updated profile
    user_profile.last_interaction = datetime.now().isoformat()
    save_session(user_profile)
    # Add sources information
    source_info = "\n\nSources consulted: " + ", ".join(set(sources))
    final_response = response + source_info
    # Save the interaction to memory
    conversation_memory.save_context(
        {"input": user_input},
        {"output": final_response}
    )
    return final_response
# -- Similarly update the other functions to use invoke instead of get_relevant_documents --
def compare_policies(user_input: str, user_profile: UserProfile, conversation_turn: int):
    """Enhanced comparison tool with profile awareness"""
    # Get conversation history
    conversation_history = conversation_memory.load_memory_variables({})
    history_text = conversation_history.get("history", "")
    # Extract policies to compare from user input or history
    policies_to_compare = []
    # First check if specific policies are mentioned in this query
    input_policies = extract_policy_entities(user_input, conversation_turn)
    if input_policies:
        policies_to_compare.extend(input_policies)
    # If fewer than 2 policies identified, look at user's history
    if len(policies_to_compare) < 2:
        # Try to find recently discussed/recommended policies
        discussed_policies = sorted(
            user_profile.policies_discussed,
            key=lambda x: x.mentioned_in_turn,
            reverse=True
        )
        # Add most recently discussed policies not already in the list
        for policy in discussed_policies:
            if len(policies_to_compare) >= 2:
                break
            # Check if already in our list
            already_added = any(p.policy_name.lower() == policy.policy_name.lower() for p in policies_to_compare)
            if not already_added:
                policies_to_compare.append(policy)
    # Format the policy list for retrieval
    policy_names = [policy.policy_name for policy in policies_to_compare]
    # Create a combined query for retrieval
    enhanced_query = user_input
    if policy_names:
        enhanced_query += " Comparing policies: " + ", ".join(policy_names)
    # Retrieve relevant documents - Fix for LangChain deprecation warning
    retriever = chroma_db.as_retriever(search_kwargs={"k": 6})
    docs = retriever.invoke(enhanced_query)  # Using invoke instead of get_relevant_documents
    if not docs:
        return ":x: No matching policy information found for comparison."
    # Rest of the function remains the same...
    # Extract context from documents
    contexts = [doc.page_content for doc in docs]
    sources = [doc.metadata.get("source", "Unknown") for doc in docs]
    # Format policy list as text for the prompt
    policy_list_text = ""
    if policy_names:
        policy_list_text = "Policies to compare:\n- " + "\n- ".join(policy_names)
    # Create a comparison prompt template with conversation history
    comparison_template = """
    You are an insurance expert for ICICI. The user wants to compare different insurance policies.
    CONVERSATION HISTORY:
    {history}
    USER QUERY: {query}
    POLICIES TO COMPARE:
    {policy_list}
    POLICY INFORMATION:
    {context}
    Please provide:
    1. A detailed comparison of the specified policies, or if none are specified, the most relevant policies from the context
    2. Compare key factors like coverage, premiums, benefits, exclusions, and claim process
    3. Create a clear comparison structure that makes differences and similarities easy to understand
    4. Provide a brief recommendation on which policy might be better for what type of user
    Your response should be organized in a clear, comparative format. Bold the policy names when you refer to them for clarity.
    """
    # Create the prompt
    prompt = ChatPromptTemplate.from_template(comparison_template)
    # Build the chain
    chain = (
        {"query": RunnablePassthrough(),
         "context": lambda x: "\n\n".join(contexts),
         "history": lambda x: history_text,
         "policy_list": lambda x: policy_list_text}
        | prompt
        | llm
        | StrOutputParser()
    )
    # Run the chain
    response = chain.invoke(user_input)
    # Extract policy entities from the response
    policies = extract_policy_entities(response, conversation_turn)
    # Add these policies to the discussed list in user profile
    for policy in policies:
        # Check if this policy is already in the discussed list
        already_discussed = False
        for existing in user_profile.policies_discussed:
            if existing.policy_name.lower() == policy.policy_name.lower():
                already_discussed = True
                break
        if not already_discussed:
            # Fact check before adding
            fact_check_result = True  # Assume true by default
            if policy.policy_name:
                # Extract the part of the response describing this policy
                policy_pattern = rf"(?:.*{re.escape(policy.policy_name)}.*(?:\n.*)*)"
                matches = re.search(policy_pattern, response)
                if matches:
                    policy_description = matches.group(0)
                    fact_check_result = fact_check_policy_info(policy.policy_name, policy_description)
            if fact_check_result:
                user_profile.policies_discussed.append(policy)
    # Save updated profile
    user_profile.last_interaction = datetime.now().isoformat()
    save_session(user_profile)
    # Add sources information
    source_info = "\n\nSources consulted: " + ", ".join(set(sources))
    final_response = response + source_info
    # Save the interaction to memory
    conversation_memory.save_context(
        {"input": user_input},
        {"output": final_response}
    )
    return final_response

# FAQ Tool with LLM and Fact Checking
def answer_faq(user_input: str, user_profile: UserProfile, conversation_turn: int):
    """Enhanced FAQ tool with fact checking"""
    # Get conversation history
    conversation_history = conversation_memory.load_memory_variables({})
    history_text = conversation_history.get("history", "")
    
    # Extract policies from user input
    mentioned_policies = extract_policy_entities(user_input, conversation_turn)
    
    # Format the policy list for retrieval
    policy_names = [policy.policy_name for policy in mentioned_policies]
    
    # Create a query that prioritizes mentioned policies
    enhanced_query = user_input
    if policy_names:
        enhanced_query += " Regarding policies: " + ", ".join(policy_names)
    
    # Retrieve relevant documents
    retriever = chroma_db.as_retriever(search_kwargs={"k": 4})
    docs = retriever.get_relevant_documents(enhanced_query)
    if not docs:
        return ":x: I don't have information about that in my policy database. Please try another question or contact ICICI customer service."
    
    # Extract context from documents
    contexts = [doc.page_content for doc in docs]
    sources = [doc.metadata.get("source", "Unknown") for doc in docs]
    
    # Create an FAQ answering prompt template with conversation history
    faq_template = """
    You are a knowledgeable insurance assistant for ICICI. Answer the user's question about insurance policies
    based on the available information and conversation history.
    
    CONVERSATION HISTORY:
    {history}
    
    USER QUESTION: {query}
    
    POLICY INFORMATION:
    {context}
    
    Please provide:
    1. A direct and clear answer to the user's question based strictly on the provided information
    2. Any relevant additional information that might be helpful
    3. If the information available is insufficient, mention what the user should do to get more information
    
    When you mention specific policies, bold their names for clarity. Only state facts that are directly supported by the provided policy information.
    """
    
    # Create the prompt
    prompt = ChatPromptTemplate.from_template(faq_template)
    
    # Build the chain
    chain = (
        {"query": RunnablePassthrough(), 
         "context": lambda x: "\n\n".join(contexts), 
         "history": lambda x: history_text}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # Run the chain
    response = chain.invoke(user_input)
    
    # Perform fact checking on the response
    facts_verified = True
    
    # Extract policy entities from the response for fact checking
    response_policies = extract_policy_entities(response, conversation_turn)
    
    # For each policy mentioned in the response, verify the facts
    for policy in response_policies:
        # Extract the part of the response describing this policy
        policy_pattern = rf"(?:.*{re.escape(policy.policy_name)}.*(?:\n.*)*)"
        matches = re.search(policy_pattern, response)
        if matches:
            policy_description = matches.group(0)
            if not fact_check_policy_info(policy.policy_name, policy_description):
                facts_verified = False
                break
    
    # If facts not verified, add a disclaimer
    if not facts_verified:
        response += "\n\nNote: Some information provided may not be complete or fully accurate based on our policy database. Please verify critical details with ICICI customer service."
    
    # Add these policies to the discussed list in user profile
    for policy in response_policies:
        # Check if this policy is already in the discussed list
        already_discussed = False
        for existing in user_profile.policies_discussed:
            if existing.policy_name.lower() == policy.policy_name.lower():
                already_discussed = True
                break
        
        if not already_discussed:
            user_profile.policies_discussed.append(policy)
    
    # Save updated profile
    user_profile.last_interaction = datetime.now().isoformat()
    save_session(user_profile)
    
    # Add sources information
    source_info = "\n\nSources consulted: " + ", ".join(set(sources))
    final_response = response + source_info

    # Save the interaction to memory
    conversation_memory.save_context(
        {"input": user_input},
        {"output": final_response}
    )
    
    return final_response

# --- Main Chat Function with Dynamic Tool Selection ---
def process_chat(user_id: str, user_input: str, conversation_turn: int):
    """Process user input and route to the appropriate tool"""
    # Get or create user profile
    user_profile = get_or_create_session(user_id)
    
    # Update last interaction timestamp
    user_profile.last_interaction = datetime.now().isoformat()
    
    # Skip tool selection for simple greetings or very short inputs
    if len(user_input.strip()) < 10 or re.match(r'^(hi|hello|hey|greetings|good (morning|afternoon|evening)).*$', user_input.lower()):
        # Just respond with a general greeting
        greeting_response = f"Hello! I'm the ICICI Insurance Assistant. How can I help you today with your insurance needs?"
        
        # Save to memory
        conversation_memory.save_context(
            {"input": user_input},
            {"output": greeting_response}
        )
        
        return greeting_response
    
    # Get embedding for user input
    input_embedding = embedding_model.embed_query(user_input)
    
    # Compare with tool embeddings
    similarities = np.dot(tool_embeddings, input_embedding)
    most_similar_idx = np.argmax(similarities)
    most_similar_tool = tool_names[most_similar_idx]
    
    # Route to appropriate tool based on similarity
    if most_similar_tool == "recommendation":
        return recommend_policy(user_input, user_profile, conversation_turn)
    elif most_similar_tool == "comparison":
        return compare_policies(user_input, user_profile, conversation_turn)
    elif most_similar_tool == "faq":
        return answer_faq(user_input, user_profile, conversation_turn)
    elif most_similar_tool == "similar_policy":
        return find_similar_policies(user_input, user_profile, conversation_turn)
    else:
        # Fallback to FAQ as default
        return answer_faq(user_input, user_profile, conversation_turn)

# --- Simple CLI for Testing ---
def run_cli():
    """Run a simple command-line interface for testing"""
    print("\n===== ICICI Insurance Assistant =====")
    print("(Type 'exit' to quit)")
    
    # Generate a random user ID for this session
    user_id = str(uuid.uuid4())
    turn_counter = 1
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'exit':
            break
            
        response = process_chat(user_id, user_input, turn_counter)
        print(f"\n {response}")
        turn_counter += 1

# --- Entry Point ---
if __name__ == "__main__":
    run_cli()