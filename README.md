# robotAiSim3D — CausalBot

A futuristic, LLM-powered 3D robot simulation framework designed to explore autonomous robotic reasoning, dynamic skill acquisition, and physics-based task execution.

## 🚀 Project Idea

**CausalBot** is more than just a 3D simulation; it is a "living" research environment where a robotic agent uses Large Language Models (LLMs) to bridge the gap between high-level natural language instructions and low-level physical actions. 

The core philosophy revolves around **Autonomous Reasoning (Chain of Thought)** and **Dynamic Skill Invention**. Instead of being hardcoded with Every possible action, the robot "thinks" through problems, plans its steps, and can even write its own JavaScript code to perform new tasks it hasn't seen before.

## 🧠 Key Features

- **Chain of Thought (CoT):** The robot mimics human reasoning by breaking down complex instructions into intermediate logical steps, visible in real-time within the UI.
- **Dynamic Skill Invention:** When faced with a task beyond its current capability, the robot uses an LLM (Gemini) to generate and "invent" new JavaScript skills, which are سپس tested and saved to a persistent **Skill Registry**.
- **Physics-Driven Execution:** Built with **Rapier3D**, the simulation ensures that every movement, collision, and object interaction follows realistic physical laws.
- **Memory & Feedback Loop:** Every action is logged in a **Memory Log**. Successes and failures inform future decisions, creating a basic learning cycle.
- **Premium 3D Visuals:** A sleek, dark-themed dashboard using **Three.js** with real-time status monitoring, eye-color state indicators, and floating UI elements.

## 🔭 Project Scope

The project currently focuses on indoor navigation and basic object manipulation within a bounded 3D space:
- **Environment:** A 3x3 bounded room with floor physics and various interactable objects (balls, boxes, etc.).
- **Intelligence:** Integration with Gemini API for planning and code generation.
- **Control:** High-level navigation (A* / Pathfinding) and low-level physical control (Arm manipulation, rotation, jumping).

## 🎯 Expected Outcome

The ultimate goal of **robotAiSim3D** is to create a fully autonomous agent capable of:
1. **Self-Correction:** Understanding why a task failed through physics feedback and re-planning.
2. **Knowledge Persistence:** Building an extensive library of "invented" skills that grow more complex over time.
3. **Natural Interaction:** Serving as a bridge for humans to interact with complex robotic systems using simple, conversational language.

## 🛠️ Tech Stack

- **Core:** JavaScript, Vite
- **3D Engine:** [Three.js](https://threejs.org/)
- **Physics Engine:** [Rapier3D](https://rapier.rs/)
- **AI Brain:** Google Gemini (2.5-flash-lite)
- **Styling:** Vanilla CSS (Glassmorphism / Neon Aesthetics)

## ⚙️ Getting Started

### Prerequisites
- Node.js (v18+)
- A Google Gemini API Key

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/amarnath3003/robotAiSim3D.git
   ```
2. Navigate to the `causalbot` directory:
   ```bash
   cd causalbot
   ```
3. Install dependencies:
   ```bash
   npm install
   ```
4. Create a `.env` file in the `causalbot` folder and add your API key:
   ```env
   VITE_GEMINI_API_KEY=your_api_key_here
   ```
5. Run the development server:
   ```bash
   npm run dev
   ```

---

*Developed with ❤️ as a playground for AI & Robotics.*
