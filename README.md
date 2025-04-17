# Docker Container Generator Agent

This project provides an intelligent agent that generates complete Docker container applications based on user requirements. It leverages OpenAI's GPT-4.1 and LangChain to analyze requirements, generate Dockerfiles, application code, and setup instructions.

## Features
- Generates Dockerfiles following best practices
- Creates application files (e.g., Python Flask, Node.js, etc.)
- Optionally generates docker-compose.yml
- Provides setup instructions and explanations
- CLI and interactive modes

## Usage

1. **Install dependencies**
   ```bash
   pip install -r requirement.txt
   ```

2. **Set your OpenAI API key**
   ```bash
   set OPENAI_API_KEY=your_openai_api_key
   ```

3. **Run the agent**
   ```bash
   python docker_agent.py "Create a Python Flask API with Redis backend"
   ```

4. **Interactive mode**
   ```bash
   python docker_agent.py --interactive
   ```

Generated files will be saved in the `docker_output/` directory by default.

## Project Structure
- `docker_agent.py` - Main agent code
- `requirement.txt` - Python dependencies
- `docker_output/` - Generated Docker project files

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

### MIT License

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
