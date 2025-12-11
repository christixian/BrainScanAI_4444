import { Info, Stethoscope, Activity, AlertCircle } from "lucide-react";

interface MedicalInfoProps {
    prediction: string;
}

const MEDICAL_DATA: Record<string, {
    description: string;
    symptoms: string[];
    causes: string[];
    treatment: string[];
}> = {
    glioma: {
        description: "Gliomas are tumors that occur in the brain and spinal cord. They begin in the gluey supportive cells (glial cells) that surround nerve cells and help them function.",
        symptoms: [
            "Headache",
            "Nausea or vomiting",
            "Confusion or decline in brain function",
            "Memory loss",
            "Personality changes or irritability",
            "Difficulty with balance",
            "Urinary incontinence",
            "Vision problems",
            "Speech difficulties"
        ],
        causes: [
            "Exact cause is unknown",
            "Genetic factors",
            "Exposure to radiation",
            "Family history of glioma"
        ],
        treatment: [
            "Surgery",
            "Radiation therapy",
            "Chemotherapy",
            "Targeted drug therapy",
            "Tumor treating fields (TTF) therapy"
        ]
    },
    meningioma: {
        description: "A meningioma is a tumor that arises from the meninges — the membranes that surround your brain and spinal cord. Most meningiomas are noncancerous (benign), though some can be cancerous.",
        symptoms: [
            "Changes in vision, such as seeing double or blurriness",
            "Headaches that are worse in the morning",
            "Hearing loss or ringing in the ears",
            "Memory loss",
            "Loss of smell",
            "Seizures",
            "Weakness in your arms or legs"
        ],
        causes: [
            "Radiation treatment",
            "Female hormones (more common in women)",
            "Inherited nervous system disorder (neurofibromatosis type 2)",
            "Obesity"
        ],
        treatment: [
            "Wait-and-see approach (for small, slow-growing tumors)",
            "Surgery",
            "Radiation therapy",
            "Medications (clinical trials)"
        ]
    },
    pituitary: {
        description: "Pituitary tumors are abnormal growths that develop in your pituitary gland. Some pituitary tumors result in too many of the hormones that regulate important functions of your body. Some pituitary tumors can cause your pituitary gland to produce lower levels of hormones.",
        symptoms: [
            "Headache",
            "Vision loss (peripheral vision)",
            "Nausea or vomiting",
            "Weakness",
            "Feeling cold",
            "Less frequent or no menstrual periods",
            "Sexual dysfunction",
            "Increased amount of urine",
            "Unintended weight loss or gain"
        ],
        causes: [
            "Cause is largely unknown",
            "Genetic mutations (Multiple Endocrine Neoplasia, type 1 - MEN1)",
            "Most cases are not inherited"
        ],
        treatment: [
            "Surgery (Transsphenoidal surgery)",
            "Radiation therapy",
            "Medications to control hormone production",
            "Replacement of missing hormones"
        ]
    },
    notumor: {
        description: "No tumor detected. The scan appears to show normal brain structure without evidence of Glioma, Meningioma, or Pituitary tumors.",
        symptoms: [],
        causes: [],
        treatment: []
    },
    
    uncertain: {
        description: "Model is not confident in the result. Please contact a medical professional.",
        symptoms: [],
        causes: [],
        treatment: []
    }
};

export default function MedicalInfo({ prediction }: MedicalInfoProps) {
    const data = MEDICAL_DATA[prediction.toLowerCase()] || MEDICAL_DATA["notumor"];

    if (prediction.toLowerCase() === "notumor") {
        return (
            <div className="mt-8 p-6 rounded-xl bg-emerald-500/10 border border-emerald-500/20 animate-fade-in">
                <div className="flex items-start gap-4">
                    <div className="p-3 rounded-lg bg-emerald-500/20 text-emerald-400">
                        <Activity className="w-6 h-6" />
                    </div>
                    <div>
                        <h3 className="text-xl font-bold text-white mb-2">Healthy Scan</h3>
                        <p className="text-slate-300 leading-relaxed">{data.description}</p>
                    </div>
                </div>
            </div>
        );
    }

    if (prediction.toLowerCase() === "uncertain") {
        return (
            <div className="mt-8 p-6 rounded-xl bg-emerald-500/10 border border-emerald-500/20 animate-fade-in">
                <div className="flex items-start gap-4">
                    <div className="p-3 rounded-lg bg-emerald-500/20 text-emerald-400">
                        <Activity className="w-6 h-6" />
                    </div>
                    <div>
                        <h3 className="text-xl font-bold text-white mb-2">Uncertain</h3>
                        <p className="text-slate-300 leading-relaxed">{data.description}</p>
                    </div>
                </div>
            </div>
        );
    }


    return (
        <div className="mt-12 space-y-8 animate-fade-in">
            <div className="flex items-center gap-3 mb-6">
                <div className="p-2 rounded-lg bg-cyan-500/20 text-cyan-400">
                    <Stethoscope className="w-6 h-6" />
                </div>
                <h2 className="text-2xl font-bold text-white">Medical Information</h2>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Description */}
                <div className="col-span-1 md:col-span-2 p-6 rounded-2xl bg-slate-800/50 border border-slate-700/50">
                    <h3 className="text-lg font-semibold text-cyan-400 mb-3 flex items-center gap-2">
                        <Info className="w-5 h-5" />
                        About {prediction}
                    </h3>
                    <p className="text-slate-300 leading-relaxed">
                        {data.description}
                    </p>
                </div>

                {/* Symptoms */}
                <div className="p-6 rounded-2xl bg-slate-800/50 border border-slate-700/50">
                    <h3 className="text-lg font-semibold text-rose-400 mb-4 flex items-center gap-2">
                        <AlertCircle className="w-5 h-5" />
                        Common Symptoms
                    </h3>
                    <ul className="space-y-2">
                        {data.symptoms.map((symptom, idx) => (
                            <li key={idx} className="flex items-start gap-2 text-slate-300 text-sm">
                                <span className="mt-1.5 w-1.5 h-1.5 rounded-full bg-rose-500/50 shrink-0" />
                                {symptom}
                            </li>
                        ))}
                    </ul>
                </div>

                {/* Treatment */}
                <div className="p-6 rounded-2xl bg-slate-800/50 border border-slate-700/50">
                    <h3 className="text-lg font-semibold text-emerald-400 mb-4 flex items-center gap-2">
                        <Activity className="w-5 h-5" />
                        Standard Treatments
                    </h3>
                    <ul className="space-y-2">
                        {data.treatment.map((item, idx) => (
                            <li key={idx} className="flex items-start gap-2 text-slate-300 text-sm">
                                <span className="mt-1.5 w-1.5 h-1.5 rounded-full bg-emerald-500/50 shrink-0" />
                                {item}
                            </li>
                        ))}
                    </ul>
                </div>
            </div>

            <div className="text-center text-xs text-slate-500 mt-8">
                <p>Source: General medical knowledge base. Information is for educational purposes only.</p>
            </div>
        </div>
    );
}
