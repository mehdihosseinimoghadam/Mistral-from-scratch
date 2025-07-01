# Mistral-from-scratch
Mistral from scratch code and math behind




graph TD
    A["Pre-trained LLM<br/>🤖"] --> B["Supervised Fine-Tuning<br/>(SFT)<br/>📚"]
    B --> C["Reference Policy<br/>π_ref(y|x)<br/>🎯"]
    
    C --> D["Reward Learning<br/>Phase<br/>🏆"]
    D --> E["Human Preference<br/>Data Collection<br/>👥"]
    E --> F["Bradley-Terry Model<br/>p(y_w ≻ y_l | x)<br/>⚖️"]
    F --> G["Reward Function<br/>r(x,y)<br/>💎"]
    
    G --> H["RL Fine-Tuning<br/>Phase<br/>🎮"]
    H --> I["KL-Regularized<br/>Optimization<br/>🔄"]
    I --> J["Final Aligned<br/>Policy π*<br/>✨"]
    
    style A fill:#FFE6E6,stroke:#FF6B6B,stroke-width:3px
    style B fill:#E6F2FF,stroke:#4A90E2,stroke-width:3px
    style C fill:#E6FFE6,stroke:#7ED321,stroke-width:3px
    style D fill:#FFF2E6,stroke:#F5A623,stroke-width:3px
    style E fill:#F0E6FF,stroke:#9013FE,stroke-width:3px
    style F fill:#FFE6F2,stroke:#E91E63,stroke-width:3px
    style G fill:#E6FFF9,stroke:#00C851,stroke-width:3px
    style H fill:#FFF9E6,stroke:#FF8A00,stroke-width:3px
    style I fill:#E6F9FF,stroke:#00BCD4,stroke-width:3px
    style J fill:#FFFFE6,stroke:#FFEB3B,stroke-width:3px
