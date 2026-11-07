
import { Link } from "react-router-dom";
import Navbar from "../components/layout/Navbar";
import {
  ArrowRight,
  FileText,
  Sparkles,
  LayoutTemplate,
  MousePointer2,
  Download,
  CheckCircle2,
  Upload,
  BrainCircuit,
  Palette,
  Layers3,
  Zap,
  Atom,
  ChartNoAxesCombined,
} from "lucide-react";

const features = [
  {
    icon: FileText,
    title: "Smart Document Analysis",
    description: "Upload research papers and organize important information into structured sections.",
  },
  {
    icon: BrainCircuit,
    title: "AI-Assisted Organization",
    description: "Prepare research content for clear and meaningful visual communication.",
  },
  {
    icon: LayoutTemplate,
    title: "Professional Templates",
    description: "Choose from scientific, engineering, academic and AI research layouts.",
  },
  {
    icon: MousePointer2,
    title: "Interactive Poster Editor",
    description: "Customize text, images, diagrams and every individual poster element.",
  },
  {
    icon: Layers3,
    title: "Complete Creative Control",
    description: "Manage layers, positioning, typography, colors and dimensions.",
  },
  {
    icon: Download,
    title: "Preview and Export",
    description: "Review your finished research poster and prepare it for export.",
  },
];

const workflow = [
  {
    number: "01",
    icon: Upload,
    title: "Upload Research",
    description: "Upload your research paper in PDF or DOCX format.",
  },
  {
    number: "02",
    icon: BrainCircuit,
    title: "Review Content",
    description: "Organize, edit and confirm important research information.",
  },
  {
    number: "03",
    icon: Palette,
    title: "Design Poster",
    description: "Select a template and customize your poster visually.",
  },
  {
    number: "04",
    icon: Download,
    title: "Preview and Export",
    description: "Review your design and prepare your final poster.",
  },
];

const templates = [
  {
    title: "Research Paper Poster",
    category: "Academic Research",
    gradient: "from-blue-600 to-indigo-800",
  },
  {
    title: "Scientific Conference",
    category: "Scientific Presentation",
    gradient: "from-violet-600 to-purple-900",
  },
  {
    title: "Engineering Project",
    category: "Technical Projects",
    gradient: "from-cyan-600 to-blue-900",
  },
  {
    title: "AI / ML Research",
    category: "Artificial Intelligence",
    gradient: "from-fuchsia-600 to-violet-900",
  },
  {
    title: "Experimental Results",
    category: "Data and Analysis",
    gradient: "from-emerald-600 to-teal-900",
  },
];

function PosterPreview() {
  return (
    <div className="relative mx-auto w-full max-w-lg">
      <div className="absolute -inset-10 rounded-full bg-blue-300/30 blur-3xl" />

      <div className="relative rounded-2xl border border-slate-200 bg-white p-6 shadow-2xl shadow-blue-950/10">
        <div className="flex items-center justify-between border-b pb-4">
          <div>
            <div className="h-2 w-24 rounded bg-blue-600" />
            <div className="mt-2 h-2 w-36 rounded bg-slate-200" />
          </div>
          <Atom className="text-blue-600" size={30} />
        </div>

        <h3 className="my-6 text-center text-xl font-extrabold text-slate-900">
          Intelligent Research & Innovation
        </h3>

        <div className="grid grid-cols-3 gap-3">
          <div className="space-y-3">
            <div className="h-3 w-3/4 rounded bg-blue-600" />
            <div className="h-24 rounded-lg bg-blue-50 p-3">
              <div className="space-y-2">
                <div className="h-2 rounded bg-blue-200" />
                <div className="h-2 rounded bg-blue-200" />
                <div className="h-2 w-2/3 rounded bg-blue-200" />
              </div>
            </div>
            <div className="h-3 w-2/3 rounded bg-blue-600" />
            <div className="h-20 rounded-lg bg-slate-100" />
          </div>

          <div className="space-y-3">
            <div className="h-3 w-3/4 rounded bg-violet-600" />
            <div className="flex h-28 items-center justify-center rounded-lg bg-violet-50">
              <ChartNoAxesCombined size={40} className="text-violet-500" />
            </div>
            <div className="h-3 w-2/3 rounded bg-violet-600" />
            <div className="h-20 rounded-lg bg-slate-100" />
          </div>

          <div className="space-y-3">
            <div className="h-3 w-3/4 rounded bg-cyan-600" />
            <div className="h-24 rounded-lg bg-cyan-50" />
            <div className="h-3 w-2/3 rounded bg-cyan-600" />
            <div className="h-20 rounded-lg bg-emerald-50" />
          </div>
        </div>

        <div className="mt-5 flex gap-2 border-t pt-4">
          <div className="h-2 flex-1 rounded bg-blue-100" />
          <div className="h-2 flex-1 rounded bg-violet-100" />
          <div className="h-2 flex-1 rounded bg-cyan-100" />
        </div>
      </div>

      <div className="absolute -right-5 top-12 hidden rounded-xl border bg-white p-3 shadow-lg sm:block">
        <div className="flex items-center gap-2">
          <CheckCircle2 className="text-emerald-500" />
          <span className="text-xs font-bold text-slate-800">
            Editable elements
          </span>
        </div>
      </div>
    </div>
  );
}

export default function LandingPage() {
  return (
    <div className="min-h-screen overflow-hidden bg-white">
      <Navbar />

      {/* HERO SECTION */}
      <section className="relative bg-gradient-to-b from-blue-50 via-white to-white">
        <div className="mx-auto grid max-w-7xl items-center gap-16 px-6 py-24 lg:grid-cols-2 lg:px-10 lg:py-32">

          <div>
            <div className="mb-7 inline-flex items-center gap-2 rounded-full border border-blue-100 bg-white px-4 py-2 text-xs font-bold text-blue-700 shadow-sm">
              <Sparkles size={15} />
              RESEARCH MEETS VISUAL STORYTELLING
            </div>

            <h1 className="text-5xl font-extrabold leading-tight tracking-tight text-slate-950 sm:text-6xl">
              Turn your research into{" "}
              <span className="bg-gradient-to-r from-blue-600 to-violet-600 bg-clip-text text-transparent">
                visual impact.
              </span>
            </h1>

            <p className="mt-7 max-w-xl text-lg leading-8 text-slate-600">
              Transform technical documents and research papers into
              professional, editable scientific posters with Infographix.
            </p>

            <div className="mt-10 flex flex-wrap gap-4">
              <Link
                to="/register"
                className="flex items-center gap-3 rounded-xl bg-blue-600 px-7 py-4 font-bold text-white shadow-lg shadow-blue-200 transition hover:-translate-y-1 hover:bg-blue-700"
              >
                Create Your Poster <ArrowRight size={19} />
              </Link>

              <a
                href="#features"
                className="rounded-xl border border-slate-200 bg-white px-7 py-4 font-bold text-slate-700 hover:border-blue-300 hover:text-blue-700"
              >
                Explore Features
              </a>
            </div>

            <div className="mt-10 flex flex-wrap gap-6 text-sm text-slate-500">
              <span className="flex items-center gap-2">
                <CheckCircle2 size={17} className="text-emerald-500" />
                Editable designs
              </span>
              <span className="flex items-center gap-2">
                <CheckCircle2 size={17} className="text-emerald-500" />
                Research focused
              </span>
            </div>
          </div>

          <PosterPreview />
        </div>
      </section>

      {/* FEATURES SECTION */}
      <section id="features" className="mx-auto max-w-7xl px-6 py-24 lg:px-10">
        <div className="mx-auto mb-14 max-w-2xl text-center">
          <span className="text-xs font-bold uppercase tracking-widest text-blue-600">
            Powerful capabilities
          </span>

          <h2 className="mt-4 text-4xl font-extrabold text-slate-950">
            Everything your research needs
          </h2>

          <p className="mt-5 leading-7 text-slate-500">
            A unified workspace for organizing research content and creating
            professional technical posters.
          </p>
        </div>

        <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
          {features.map((feature) => {
            const Icon = feature.icon;

            return (
              <article
                key={feature.title}
                className="group rounded-2xl border border-slate-200 bg-white p-7 transition hover:-translate-y-1 hover:border-blue-200 hover:shadow-xl"
              >
                <div className="mb-6 flex h-12 w-12 items-center justify-center rounded-xl bg-blue-50 text-blue-600">
                  <Icon size={23} />
                </div>

                <h3 className="text-lg font-bold text-slate-900">
                  {feature.title}
                </h3>

                <p className="mt-3 text-sm leading-7 text-slate-500">
                  {feature.description}
                </p>
              </article>
            );
          })}
        </div>
      </section>

      {/* WORKFLOW SECTION */}
      <section id="workflow" className="bg-slate-950 px-6 py-24 text-white">
        <div className="mx-auto max-w-7xl">
          <div className="mx-auto mb-16 max-w-2xl text-center">
            <span className="text-xs font-bold uppercase tracking-widest text-blue-400">
              Simple workflow
            </span>

            <h2 className="mt-4 text-4xl font-extrabold">
              From document to poster
            </h2>

            <p className="mt-5 leading-7 text-slate-400">
              Four simple steps to organize your research and create
              professional visual presentations.
            </p>
          </div>

          <div className="grid gap-8 sm:grid-cols-2 lg:grid-cols-4">
            {workflow.map((step) => {
              const Icon = step.icon;

              return (
                <div
                  key={step.number}
                  className="rounded-2xl border border-white/10 bg-white/[0.04] p-7"
                >
                  <div className="flex items-center justify-between">
                    <div className="rounded-xl bg-blue-500/15 p-3 text-blue-400">
                      <Icon size={24} />
                    </div>
                    <span className="text-3xl font-black text-white/20">
                      {step.number}
                    </span>
                  </div>

                  <h3 className="mt-8 text-lg font-bold">{step.title}</h3>

                  <p className="mt-3 text-sm leading-7 text-slate-400">
                    {step.description}
                  </p>
                </div>
              );
            })}
          </div>
        </div>
      </section>

      <section className="border-y border-slate-100 bg-slate-50/80 px-6 py-14">
        <div className="mx-auto flex max-w-7xl flex-col items-center justify-between gap-6 sm:flex-row lg:px-10">
          <div>
            <span className="text-xs font-bold uppercase tracking-widest text-blue-600">Bring your research</span>
            <h2 className="mt-2 text-2xl font-extrabold text-slate-950">Works with the formats you already use.</h2>
            <p className="mt-2 text-sm text-slate-500">Start with a paper or create a poster directly from your own ideas.</p>
          </div>
          <div className="flex items-center gap-3">
            <span className="rounded-xl border border-slate-200 bg-white px-5 py-3 text-sm font-bold text-slate-700 shadow-sm">PDF <small className="ml-2 font-medium text-slate-400">.pdf</small></span>
            <span className="rounded-xl border border-slate-200 bg-white px-5 py-3 text-sm font-bold text-slate-700 shadow-sm">Word <small className="ml-2 font-medium text-slate-400">.docx</small></span>
          </div>
        </div>
      </section>

      {/* TEMPLATES SECTION */}
      <section id="templates" className="mx-auto max-w-7xl px-6 py-24 lg:px-10">
        <div className="mb-12">
          <span className="text-xs font-bold uppercase tracking-widest text-blue-600">
            Design library
          </span>

          <h2 className="mt-3 text-4xl font-extrabold text-slate-950">
            Templates for every idea
          </h2>

          <p className="mt-4 text-slate-500">
            Choose a starting point for your next research presentation.
          </p>
        </div>

        <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-5">
          {templates.map((template) => (
            <article
              key={template.title}
              className="group overflow-hidden rounded-2xl border border-slate-200 bg-white transition hover:-translate-y-1 hover:shadow-xl"
            >
              <div className={`flex aspect-[4/5] items-center justify-center bg-gradient-to-br ${template.gradient} p-5`}>
                <div className="w-full rounded-lg bg-white p-4 shadow-xl">
                  <div className="h-2 w-3/4 rounded bg-slate-800" />
                  <div className="mt-2 h-1 w-1/2 rounded bg-slate-300" />

                  <div className="mt-5 grid grid-cols-2 gap-2">
                    <div className="h-16 rounded bg-blue-100" />
                    <div className="h-16 rounded bg-violet-100" />
                    <div className="h-12 rounded bg-slate-100" />
                    <div className="h-12 rounded bg-cyan-100" />
                  </div>

                  <div className="mt-3 h-3 rounded bg-slate-100" />
                </div>
              </div>

              <div className="p-4">
                <h3 className="text-sm font-bold text-slate-900">
                  {template.title}
                </h3>
                <p className="mt-1 text-xs text-slate-500">
                  {template.category}
                </p>
              </div>
            </article>
          ))}
        </div>
      </section>

      {/* CTA SECTION */}
      <section id="about" className="px-6 pb-24">
        <div className="mx-auto max-w-7xl rounded-3xl bg-gradient-to-r from-blue-700 via-indigo-700 to-violet-700 px-8 py-16 text-center text-white shadow-2xl lg:px-20">
          <Zap size={36} className="mx-auto mb-6" />

          <h2 className="text-3xl font-extrabold sm:text-4xl">
            Give your research a visual identity.
          </h2>

          <p className="mx-auto mt-5 max-w-2xl leading-7 text-blue-100">
            Bring your technical ideas together and create posters that
            communicate your work clearly.
          </p>

          <Link
            to="/register"
            className="mt-8 inline-flex items-center gap-3 rounded-xl bg-white px-7 py-4 font-bold text-blue-700 transition hover:bg-blue-50"
          >
            Get Started <ArrowRight size={18} />
          </Link>
        </div>
      </section>

      {/* FOOTER */}
      <footer className="border-t border-slate-200 bg-white">
        <div className="mx-auto flex max-w-7xl flex-col justify-between gap-6 px-6 py-10 sm:flex-row sm:items-center lg:px-10">
          <div className="flex items-center gap-3">
            <Atom size={25} className="text-blue-600" />
            <div>
              <p className="font-extrabold text-slate-900">Infographix</p>
              <p className="text-xs text-slate-500">Research to Visuals</p>
            </div>
          </div>

          <p className="text-sm text-slate-500">
            © {new Date().getFullYear()} Infographix. All rights reserved.
          </p>

          <div className="flex gap-6 text-sm font-medium text-slate-500">
            <Link to="/login">Login</Link>
            <Link to="/register">Register</Link>
          </div>
        </div>
      </footer>
    </div>
  );
}
