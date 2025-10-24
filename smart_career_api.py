from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
from pathlib import Path
import uvicorn

app = FastAPI(title="Smart Career API", version="2.0.0")

class CareerPlanRequest(BaseModel):
    current_skills: List[str]
    target_role: Optional[str] = None

class SalaryRequest(BaseModel):
    years_experience: int
    skills: List[str] = []
    currency: str = "USD"
    period: str = "YEARLY"

# Enhanced skills database with role-specific and progression data
SKILLS_DATABASE = {
    # iOS Development Skills
    "ios_skills": {
        "core": ["swift", "objective-c", "xcode", "ios frameworks", "cocoa touch"],
        "ui": ["swiftui", "uikit", "storyboard", "autolayout", "core animation"],
        "data": ["core data", "sqlite", "realm", "firebase", "cloudkit"],
        "networking": ["urlsession", "alamofire", "rest api", "json", "http"],
        "testing": ["xctest", "unit testing", "ui testing", "testflight"],
        "advanced": ["core ml", "arkit", "widgetkit", "swift concurrency", "combine"]
    },
    
    # Data Science Skills
    "data_science_skills": {
        "programming": ["python", "r", "sql", "julia", "scala"],
        "ml_frameworks": ["tensorflow", "pytorch", "scikit-learn", "keras", "xgboost"],
        "data_tools": ["pandas", "numpy", "matplotlib", "seaborn", "plotly"],
        "databases": ["postgresql", "mongodb", "redis", "elasticsearch", "hadoop"],
        "cloud": ["aws", "gcp", "azure", "docker", "kubernetes"],
        "visualization": ["tableau", "power bi", "d3.js", "jupyter", "notebooks"]
    },
    
    # Web Development Skills
    "web_dev_skills": {
        "frontend": ["javascript", "react", "vue", "angular", "html", "css"],
        "backend": ["node.js", "python", "java", "php", "ruby", "go"],
        "databases": ["mysql", "postgresql", "mongodb", "redis"],
        "cloud": ["aws", "azure", "gcp", "heroku", "vercel"],
        "tools": ["git", "docker", "kubernetes", "jenkins", "webpack"]
    },
    
    # DevOps Skills
    "devops_skills": {
        "containers": ["docker", "kubernetes", "podman", "containerd"],
        "cloud": ["aws", "azure", "gcp", "terraform", "ansible"],
        "ci_cd": ["jenkins", "gitlab ci", "github actions", "circleci"],
        "monitoring": ["prometheus", "grafana", "elk stack", "datadog"],
        "infrastructure": ["linux", "bash", "python", "terraform", "cloudformation"]
    }
}

# Skill difficulty and learning time estimates
SKILL_DIFFICULTY = {
    "beginner": ["html", "css", "excel", "git", "linux"],
    "intermediate": ["javascript", "python", "react", "docker", "sql"],
    "advanced": ["kubernetes", "machine learning", "tensorflow", "pytorch", "arkit"],
    "expert": ["distributed systems", "ai research", "blockchain", "quantum computing"]
}

# Role-specific skill priorities
ROLE_PRIORITIES = {
    "ios engineer": {
        "must_have": ["swift", "ios frameworks", "xcode", "uikit", "swiftui"],
        "should_have": ["core data", "networking", "testing", "git", "rest api"],
        "nice_to_have": ["core ml", "arkit", "widgetkit", "combine", "swift concurrency"]
    },
    "data scientist": {
        "must_have": ["python", "sql", "pandas", "numpy", "machine learning"],
        "should_have": ["tensorflow", "pytorch", "jupyter", "aws", "docker"],
        "nice_to_have": ["kubernetes", "spark", "kafka", "tableau", "power bi"]
    },
    "web developer": {
        "must_have": ["javascript", "html", "css", "react", "git"],
        "should_have": ["node.js", "sql", "aws", "docker", "rest api"],
        "nice_to_have": ["kubernetes", "microservices", "graphql", "typescript", "next.js"]
    },
    "devops engineer": {
        "must_have": ["docker", "kubernetes", "aws", "linux", "git"],
        "should_have": ["terraform", "jenkins", "python", "monitoring", "ci/cd"],
        "nice_to_have": ["ansible", "prometheus", "grafana", "helm", "istio"]
    }
}

def load_skills_data():
    skills_path = Path("data/gold/skills_demand.parquet")
    if skills_path.exists():
        try:
            return pd.read_parquet(skills_path)
        except Exception:
            pass
    return pd.DataFrame()

def get_role_skills(target_role: str) -> dict:
    """Get role-specific skills based on target role"""
    role_lower = target_role.lower() if target_role else ""
    
    if "ios" in role_lower or "mobile" in role_lower:
        return SKILLS_DATABASE["ios_skills"]
    elif "data" in role_lower or "scientist" in role_lower or "analyst" in role_lower:
        return SKILLS_DATABASE["data_science_skills"]
    elif "web" in role_lower or "frontend" in role_lower or "backend" in role_lower:
        return SKILLS_DATABASE["web_dev_skills"]
    elif "devops" in role_lower or "sre" in role_lower or "infrastructure" in role_lower:
        return SKILLS_DATABASE["devops_skills"]
    else:
        # Default to general tech skills
        return {
            "programming": ["python", "javascript", "java", "go", "rust"],
            "tools": ["git", "docker", "kubernetes", "aws", "linux"],
            "databases": ["sql", "postgresql", "mongodb", "redis"],
            "frameworks": ["react", "node.js", "spring", "django", "flask"]
        }

def calculate_skill_priority(skill: str, current_skills: List[str], target_role: str, role_skills: dict) -> float:
    """Calculate priority score for a skill based on multiple factors"""
    score = 0.0
    
    # MASSIVE bonus for role-specific skills (this is the key fix!)
    is_role_specific = False
    for category, skills in role_skills.items():
        if skill.lower() in [s.lower() for s in skills]:
            is_role_specific = True
            if category in ["core", "must_have", "programming"]:
                score += 1000  # HUGE bonus for core role skills
            elif category in ["ui", "data", "frontend", "backend", "networking", "testing"]:
                score += 800   # High bonus for important role skills
            elif category in ["advanced", "tools", "frameworks"]:
                score += 600   # Good bonus for advanced role skills
            else:
                score += 400   # Decent bonus for other role skills
    
    # Base demand score (from our skills database) - only if not role-specific
    if not is_role_specific:
        skills_df = load_skills_data()
        if not skills_df.empty and "skill" in skills_df.columns:
            skill_row = skills_df[skills_df["skill"].str.lower() == skill.lower()]
            if not skill_row.empty:
                score += float(skill_row["count"].iloc[0]) / 100  # Much lower weight for generic skills
    
    # Skill progression bonus (skills that build on current skills)
    for current_skill in current_skills:
        if current_skill.lower() in skill.lower() or skill.lower() in current_skill.lower():
            score += 50  # Higher bonus for skill progression
    
    # Role priority bonus (check against role priorities)
    if target_role and target_role.lower() in ROLE_PRIORITIES:
        role_priorities = ROLE_PRIORITIES[target_role.lower()]
        if skill.lower() in [s.lower() for s in role_priorities.get("must_have", [])]:
            score += 2000  # MASSIVE bonus for must-have skills
        elif skill.lower() in [s.lower() for s in role_priorities.get("should_have", [])]:
            score += 1000  # High bonus for should-have skills
        elif skill.lower() in [s.lower() for s in role_priorities.get("nice_to_have", [])]:
            score += 500   # Decent bonus for nice-to-have skills
    
    # Difficulty adjustment (easier skills get slight boost for learning path)
    if skill.lower() in SKILL_DIFFICULTY["beginner"]:
        score += 10
    elif skill.lower() in SKILL_DIFFICULTY["advanced"]:
        score -= 5  # Smaller penalty for advanced skills
    
    return score

def get_personalized_recommendations(current_skills: List[str], target_role: str) -> dict:
    """Generate personalized skill recommendations"""
    current_skills_lower = [s.lower().strip() for s in current_skills]
    role_skills = get_role_skills(target_role)
    
    # Get all possible skills from role categories FIRST (prioritize role-specific)
    all_role_skills = []
    for category, skills in role_skills.items():
        all_role_skills.extend(skills)
    
    # Add general tech skills as fallback
    skills_df = load_skills_data()
    if not skills_df.empty and "skill" in skills_df.columns:
        general_skills = skills_df["skill"].str.lower().tolist()
        # Only add general skills that aren't already in role skills
        for skill in general_skills:
            if skill not in [s.lower() for s in all_role_skills]:
                all_role_skills.append(skill)
    
    # Remove duplicates and current skills
    unique_skills = list(set([s.lower() for s in all_role_skills]))
    candidate_skills = [s for s in unique_skills if s not in current_skills_lower]
    
    # Calculate priority scores with MUCH higher weight for role-specific skills
    skill_scores = []
    for skill in candidate_skills:
        priority = calculate_skill_priority(skill, current_skills_lower, target_role, role_skills)
        skill_scores.append((skill, priority))
    
    # Sort by priority and get top recommendations
    skill_scores.sort(key=lambda x: x[1], reverse=True)
    top_skills = [skill for skill, score in skill_scores[:20]]
    
    # Categorize recommendations based on role priorities
    recommendations = {
        "must_learn": [],
        "should_learn": [],
        "nice_to_have": []
    }
    
    # Get role priorities for categorization
    role_priorities = {}
    if target_role:
        role_lower = target_role.lower()
        for role_key, priorities in ROLE_PRIORITIES.items():
            if role_lower in role_key or role_key in role_lower:
                role_priorities = priorities
                break
    
    for skill in top_skills[:15]:  # Top 15 recommendations
        if role_priorities:
            must_have = [s.lower() for s in role_priorities.get("must_have", [])]
            should_have = [s.lower() for s in role_priorities.get("should_have", [])]
            nice_to_have = [s.lower() for s in role_priorities.get("nice_to_have", [])]
            
            if skill in must_have:
                recommendations["must_learn"].append(skill)
            elif skill in should_have:
                recommendations["should_learn"].append(skill)
            elif skill in nice_to_have:
                recommendations["nice_to_have"].append(skill)
            else:
                # Check if it's a role-specific skill
                is_role_specific = any(skill in [s.lower() for s in skills] for skills in role_skills.values())
                if is_role_specific:
                    recommendations["should_learn"].append(skill)
                else:
                    recommendations["nice_to_have"].append(skill)
        else:
            recommendations["should_learn"].append(skill)
    
    return {
        "target_role": target_role,
        "recommended_skills": top_skills,
        "categorized_recommendations": recommendations,
        "total_skills_analyzed": len(candidate_skills),
        "your_skills": current_skills_lower,
        "learning_path": {
            "immediate_focus": recommendations["must_learn"][:5],
            "next_phase": recommendations["should_learn"][:5],
            "future_goals": recommendations["nice_to_have"][:5]
        }
    }

@app.on_event("startup")
async def startup_event():
    # Pre-load skills data for faster responses
    load_skills_data()
    print("Smart Career API started successfully!")

@app.get("/health")
def health():
    return {"ok": True, "status": "smart career api running"}

@app.post("/career-recommendations")
def career_recommendations(req: CareerPlanRequest):
    return get_personalized_recommendations(req.current_skills, req.target_role)

@app.post("/predict-salary")
def predict_salary(req: SalaryRequest):
    # Enhanced salary prediction based on skills and role
    base_salary = 50000
    
    # Experience multiplier
    experience_multiplier = 1 + (req.years_experience * 0.1)
    
    # Skills bonus
    skill_bonus = 0
    for skill in req.skills:
        if skill.lower() in ["python", "javascript", "java", "swift"]:
            skill_bonus += 5000
        elif skill.lower() in ["kubernetes", "tensorflow", "pytorch", "arkit"]:
            skill_bonus += 8000
        elif skill.lower() in ["aws", "docker", "react", "ios frameworks"]:
            skill_bonus += 3000
    
    predicted = (base_salary * experience_multiplier) + skill_bonus
    
    return {
        "predicted_salary": int(predicted),
        "lower_bound": int(predicted * 0.8),
        "upper_bound": int(predicted * 1.2),
        "currency": req.currency,
        "period": req.period,
        "skill_bonus": skill_bonus
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
