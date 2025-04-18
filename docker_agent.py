<<<<<<< HEAD
#!/usr/bin/env python3
"""
Docker Container Generator Agent

This script creates Docker container applications based on user queries using OpenAI's GPT-4.1.
It handles parsing user requirements, generating appropriate Dockerfiles and application setups.
"""

import os
import argparse
import sys
from typing import Dict, List, Optional, Union
import json
from pathlib import Path
import logging
from dotenv import load_dotenv

from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.chains import LLMChain
from langchain_community.callbacks.manager import get_openai_callback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('docker-agent')

# Load environment variables
load_dotenv()

class DockerAgent:
    """Agent that generates Docker container applications based on user requirements."""
    
    def __init__(self, model_name: str = "openai/gpt-4.1", temperature: float = 0.1, verbose: bool = False):
        """
        Initialize the Docker Agent.
        
        Args:
            model_name: The OpenAI model to use
            temperature: The temperature setting for the model
            verbose: Whether to show verbose output
        """
        # Setup OpenAI client
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        self.endpoint = os.environ.get("OPENAI_API_ENDPOINT", "https://models.github.ai/inference")
        self.model_name = model_name
        self.temperature = temperature
        self.verbose = verbose
        
        # Initialize LangChain OpenAI client
        self.llm = ChatOpenAI(
            base_url=self.endpoint,
            api_key=self.api_key,
            model_name=self.model_name,
            temperature=self.temperature
        )
        
        # Define the output schema for structured responses
        self.output_parser = self._create_output_parser()
        
        # Setup the system prompt
        self.system_prompt = self._create_system_prompt()
        
        # Setup the RunnableSequence chain (new API)
        self.chain = self._create_chain()
    
    def _create_output_parser(self):
        """Create a structured output parser for Docker container components."""
        schemas = [
            ResponseSchema(name="dockerfile", description="The complete Dockerfile content"),
            ResponseSchema(name="app_files", description="A list of dictionaries with 'filename' and 'content' keys for each application file"),
            ResponseSchema(name="docker_compose", description="Optional docker-compose.yml file if needed"),
            ResponseSchema(name="setup_instructions", description="Step-by-step instructions to set up and run the Docker container"),
            ResponseSchema(name="explanation", description="Explanation of the Docker setup and how it fulfills the requirements")
        ]
        return StructuredOutputParser.from_response_schemas(schemas)
    
    def _create_system_prompt(self):
        """Create the system prompt for the Docker container generator."""
        format_instructions = self.output_parser.get_format_instructions()
        # Escape curly braces for ChatPromptTemplate
        format_instructions = format_instructions.replace("{", "{{").replace("}", "}}")
        return f"""You are an expert Docker container developer. Your task is to create complete Docker container applications based on user requirements.

You will be given a description of what the user wants to containerize. You should:
1. Analyze the requirements carefully
2. Create a suitable Dockerfile that follows best practices
3. Generate any necessary application files
4. Provide setup instructions
5. Explain your choices

When creating the Dockerfile:
- Use the most appropriate base image
- Follow Docker best practices (minimize layers, use multi-stage builds when appropriate)
- Include proper environment variables
- Set up appropriate users (avoid running as root when possible)
- Configure correct ports and volumes
- Include proper health checks
- Optimize for security and performance

{format_instructions}

For the app_files, create a list of dictionaries, each containing the filename and content of each application file needed.
If a docker-compose.yml is needed, include it in the docker_compose field.
"""
    
    def _create_chain(self):
        """Create the RunnableSequence for processing user queries (new API)."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "{query}")
        ])
        # RunnableSequence: prompt | llm | output_parser
        return prompt | self.llm | self.output_parser

    def process_query(self, query: str) -> Dict:
        """
        Process a user query and generate Docker container application.
        
        Args:
            query: The user's requirements for the Docker container
        
        Returns:
            A dictionary containing the generated Dockerfile, application files, and instructions
        """
        logger.info(f"Processing query: {query}")
        
        try:
            with get_openai_callback() as cb:
                # Use invoke instead of run (new API)
                response = self.chain.invoke({"query": query})
                result = response  # Already parsed by output_parser
                
                if self.verbose:
                    logger.info(f"OpenAI API usage: {cb}")
                
                return result
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise
    
    def save_output(self, output: Dict, output_dir: str = "docker_output"):
        """
        Save the generated Docker container files to disk.
        
        Args:
            output: The structured output from process_query
            output_dir: The directory to save the files to
        """
        # Create output directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save Dockerfile
        with open(output_path / "Dockerfile", "w") as f:
            f.write(output["dockerfile"])
        
        # Save application files
        for file_info in output["app_files"]:
            # Create subdirectories if needed
            file_path = output_path / file_info["filename"]
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, "w") as f:
                f.write(file_info["content"])
        
        # Save docker-compose.yml if present
        if output.get("docker_compose"):
            with open(output_path / "docker-compose.yml", "w") as f:
                f.write(output["docker_compose"])
        
        # Save setup instructions and explanation
        with open(output_path / "README.md", "w") as f:
            f.write("# Docker Container Setup Instructions\n\n")
            f.write(output["setup_instructions"])
            f.write("\n\n## Explanation\n\n")
            f.write(output["explanation"])
        
        logger.info(f"Docker container files saved to {output_path.absolute()}")
        
        return output_path.absolute()


def main():
    """Main function for CLI usage."""
    parser = argparse.ArgumentParser(description="Generate Docker container applications from user requirements")
    parser.add_argument("query", nargs="?", help="The user requirements for the Docker container")
    parser.add_argument("--model", default="openai/gpt-4.1", help="The OpenAI model to use")
    parser.add_argument("--temperature", type=float, default=0.1, help="The temperature setting for the model")
    parser.add_argument("--output-dir", default="docker_output", help="Directory to save the generated files")
    parser.add_argument("--verbose", action="store_true", help="Show verbose output")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    
    args = parser.parse_args()
    
    try:
        # Create the Docker agent
        agent = DockerAgent(
            model_name=args.model,
            temperature=args.temperature,
            verbose=args.verbose
        )
        
        if args.interactive:
            print("Docker Container Generator Agent (Interactive Mode)")
            print("Type 'exit' to quit")
            print("-" * 50)
            
            while True:
                query = input("\nEnter your requirements: ")
                if query.lower() == "exit":
                    break
                
                output = agent.process_query(query)
                output_path = agent.save_output(output, args.output_dir)
                
                print(f"\nDocker container files generated in: {output_path}")
                print("\nSetup Instructions:")
                print("-" * 50)
                print(output["setup_instructions"])
        else:
            # Process the query from command line argument
            if not args.query:
                parser.print_help()
                sys.exit(1)
            
            output = agent.process_query(args.query)
            output_path = agent.save_output(output, args.output_dir)
            
            print(f"Docker container files generated in: {output_path}")
            print("\nSetup Instructions:")
            print("-" * 50)
            print(output["setup_instructions"])
    
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
=======
#!/usr/bin/env python3
"""
Docker Container Generator Agent

This script creates Docker container applications based on user queries using OpenAI's GPT-4.1.
It handles parsing user requirements, generating appropriate Dockerfiles and application setups.
"""

import os
import argparse
import sys
from typing import Dict, List, Optional, Union
import json
from pathlib import Path
import logging
from dotenv import load_dotenv

from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.chains import LLMChain
from langchain_community.callbacks.manager import get_openai_callback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('docker-agent')

# Load environment variables
load_dotenv()

class DockerAgent:
    """Agent that generates Docker container applications based on user requirements."""
    
    def __init__(self, model_name: str = "openai/gpt-4.1", temperature: float = 0.1, verbose: bool = False):
        """
        Initialize the Docker Agent.
        
        Args:
            model_name: The OpenAI model to use
            temperature: The temperature setting for the model
            verbose: Whether to show verbose output
        """
        # Setup OpenAI client
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        self.endpoint = os.environ.get("OPENAI_API_ENDPOINT", "https://models.github.ai/inference")
        self.model_name = model_name
        self.temperature = temperature
        self.verbose = verbose
        
        # Initialize LangChain OpenAI client
        self.llm = ChatOpenAI(
            base_url=self.endpoint,
            api_key=self.api_key,
            model_name=self.model_name,
            temperature=self.temperature
        )
        
        # Define the output schema for structured responses
        self.output_parser = self._create_output_parser()
        
        # Setup the system prompt
        self.system_prompt = self._create_system_prompt()
        
        # Setup the RunnableSequence chain (new API)
        self.chain = self._create_chain()
    
    def _create_output_parser(self):
        """Create a structured output parser for Docker container components."""
        schemas = [
            ResponseSchema(name="dockerfile", description="The complete Dockerfile content"),
            ResponseSchema(name="app_files", description="A list of dictionaries with 'filename' and 'content' keys for each application file"),
            ResponseSchema(name="docker_compose", description="Optional docker-compose.yml file if needed"),
            ResponseSchema(name="setup_instructions", description="Step-by-step instructions to set up and run the Docker container"),
            ResponseSchema(name="explanation", description="Explanation of the Docker setup and how it fulfills the requirements")
        ]
        return StructuredOutputParser.from_response_schemas(schemas)
    
    def _create_system_prompt(self):
        """Create the system prompt for the Docker container generator."""
        format_instructions = self.output_parser.get_format_instructions()
        # Escape curly braces for ChatPromptTemplate
        format_instructions = format_instructions.replace("{", "{{").replace("}", "}}")
        return f"""You are an expert Docker container developer. Your task is to create complete Docker container applications based on user requirements.

You will be given a description of what the user wants to containerize. You should:
1. Analyze the requirements carefully
2. Create a suitable Dockerfile that follows best practices
3. Generate any necessary application files
4. Provide setup instructions
5. Explain your choices

When creating the Dockerfile:
- Use the most appropriate base image
- Follow Docker best practices (minimize layers, use multi-stage builds when appropriate)
- Include proper environment variables
- Set up appropriate users (avoid running as root when possible)
- Configure correct ports and volumes
- Include proper health checks
- Optimize for security and performance

{format_instructions}

For the app_files, create a list of dictionaries, each containing the filename and content of each application file needed.
If a docker-compose.yml is needed, include it in the docker_compose field.
"""
    
    def _create_chain(self):
        """Create the RunnableSequence for processing user queries (new API)."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "{query}")
        ])
        # RunnableSequence: prompt | llm | output_parser
        return prompt | self.llm | self.output_parser

    def process_query(self, query: str) -> Dict:
        """
        Process a user query and generate Docker container application.
        
        Args:
            query: The user's requirements for the Docker container
        
        Returns:
            A dictionary containing the generated Dockerfile, application files, and instructions
        """
        logger.info(f"Processing query: {query}")
        
        try:
            with get_openai_callback() as cb:
                # Use invoke instead of run (new API)
                response = self.chain.invoke({"query": query})
                result = response  # Already parsed by output_parser
                
                if self.verbose:
                    logger.info(f"OpenAI API usage: {cb}")
                
                return result
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise
    
    def save_output(self, output: Dict, output_dir: str = "docker_output"):
        """
        Save the generated Docker container files to disk.
        
        Args:
            output: The structured output from process_query
            output_dir: The directory to save the files to
        """
        # Create output directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save Dockerfile
        with open(output_path / "Dockerfile", "w") as f:
            f.write(output["dockerfile"])
        
        # Save application files
        for file_info in output["app_files"]:
            # Create subdirectories if needed
            file_path = output_path / file_info["filename"]
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, "w") as f:
                f.write(file_info["content"])
        
        # Save docker-compose.yml if present
        if output.get("docker_compose"):
            with open(output_path / "docker-compose.yml", "w") as f:
                f.write(output["docker_compose"])
        
        # Save setup instructions and explanation
        with open(output_path / "README.md", "w") as f:
            f.write("# Docker Container Setup Instructions\n\n")
            f.write(output["setup_instructions"])
            f.write("\n\n## Explanation\n\n")
            f.write(output["explanation"])
        
        logger.info(f"Docker container files saved to {output_path.absolute()}")
        
        return output_path.absolute()


def main():
    """Main function for CLI usage."""
    parser = argparse.ArgumentParser(description="Generate Docker container applications from user requirements")
    parser.add_argument("query", nargs="?", help="The user requirements for the Docker container")
    parser.add_argument("--model", default="openai/gpt-4.1", help="The OpenAI model to use")
    parser.add_argument("--temperature", type=float, default=0.1, help="The temperature setting for the model")
    parser.add_argument("--output-dir", default="docker_output", help="Directory to save the generated files")
    parser.add_argument("--verbose", action="store_true", help="Show verbose output")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    
    args = parser.parse_args()
    
    try:
        # Create the Docker agent
        agent = DockerAgent(
            model_name=args.model,
            temperature=args.temperature,
            verbose=args.verbose
        )
        
        if args.interactive:
            print("Docker Container Generator Agent (Interactive Mode)")
            print("Type 'exit' to quit")
            print("-" * 50)
            
            while True:
                query = input("\nEnter your requirements: ")
                if query.lower() == "exit":
                    break
                
                output = agent.process_query(query)
                output_path = agent.save_output(output, args.output_dir)
                
                print(f"\nDocker container files generated in: {output_path}")
                print("\nSetup Instructions:")
                print("-" * 50)
                print(output["setup_instructions"])
        else:
            # Process the query from command line argument
            if not args.query:
                parser.print_help()
                sys.exit(1)
            
            output = agent.process_query(args.query)
            output_path = agent.save_output(output, args.output_dir)
            
            print(f"Docker container files generated in: {output_path}")
            print("\nSetup Instructions:")
            print("-" * 50)
            print(output["setup_instructions"])
    
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
>>>>>>> d4542fa (files added)
    main()