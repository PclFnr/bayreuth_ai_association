import { Fragment } from 'react'
import Image from 'next/image'
import clsx from 'clsx'
import { Highlight, themes } from 'prism-react-renderer'


import { Button } from '@/components/Button'
import { HeroBackground } from '@/components/HeroBackground'

const codeLanguage = 'python'
const code = 
`from skllm.models.gpt.classification.zero_shot import ZeroShotGPTClassifier

clf = ZeroShotGPTClassifier(model="gpt-4-turbo")
clf.fit(X,y)
labels = clf.predict(X)`

const tabs = [
  { name: 'app.py', isActive: true },
  // { name: 'build.py', isActive: false },
]

function TrafficLightsIcon(props) {
  return (
    <svg aria-hidden="true" viewBox="0 0 42 10" fill="none" {...props}>
      <circle cx="5" cy="5" r="4.5" />
      <circle cx="21" cy="5" r="4.5" />
      <circle cx="37" cy="5" r="4.5" />
    </svg>
  )
}

export function Hero() {
  // Function to handle scroll up by 10 pixels
  const handleScrollToElement = (event) => {
    event.preventDefault(); // Prevent default anchor link behavior
    const targetId = event.currentTarget.getAttribute('href').substring(1); // Extract the target id from href
    const targetElement = document.getElementById(targetId);

    if (targetElement) {
      const targetPosition = targetElement.getBoundingClientRect().top + window.scrollY - 20;
      window.scrollTo({ top: targetPosition, behavior: 'smooth' });
    }
  };
  return (
    // COMMENT: THE WHOLE HEADER BACKGROUND 
    <div className="overflow-hidden"> 
      <div className="py-16 sm:px-2 lg:relative lg:px-0 lg:py-20">
        <div className="mx-auto grid max-w-2xl grid-cols-1 items-center gap-x-8 gap-y-16 px-4 lg:max-w-8xl lg:grid-cols-2 lg:px-8 xl:gap-x-16 xl:px-12">
          <div className="relative z-10 md:text-center lg:text-left">
            <div className="relative">
              {/* GRADIENT HEADER SLOGAN */}
              <p className="inline text-teal-800 font-display text-5xl tracking-tight dark:text-teal-200">
              scikit-ollama
              </p>
              <p className="mt-3 text-2xl tracking-tight">
              Leverage the power of Scikit-LLM and the security of self-hosted LLMs for advanced NLP.
              </p>
              <div className="mt-8 flex gap-4 md:justify-center lg:justify-start">
                <Button href="#cde" onClick={handleScrollToElement}>
                  Get started
                </Button>
                <Button href="https://github.com/AndreasKarasenko/scikit-ollama" variant="secondary">
                  View on GitHub
                </Button>
              </div>
            </div>
          </div>
          <div>
            <li>Use self-hosted LLMs for NLP tasks</li>
            <li>Scikit-Learn compatible</li>
            <li>Built on Scikit-LLM and Ollama</li>
            <li>Open source, commercially usable - MIT License</li>
          </div>
        </div>
      </div>
    </div>
  )
}
