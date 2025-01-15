'use client'

import { usePathname } from 'next/navigation'

import { navigation } from '@/lib/navigation'

export function DocsHeader({ title }) {
  let pathname = usePathname()
  let section = navigation.find((section) =>
    section.links.find((link) => link.href === pathname),
  )

  if (!title && !section) {
    return null
  }

  return (
    <header className="mb-9 space-y-1">
      {section && (
        // COMMENT: MIDDLE SCREEN HEADER (EG INTRODUCTION)
        <p className="font-display text-sm font-medium text-teal-800 dark:text-teal-200">
          {section.title}
        </p>
      )}
      {title && (
        // COMMENT: MIDDLE SCREEN SUBHEADER (EG Getting started)
        <h1 className="font-display text-3xl tracking-tight text-zinc-900 dark:text-white">
          {title}
        </h1>
      )}
    </header>
  )
}
