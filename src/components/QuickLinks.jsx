import Link from 'next/link'

import { Icon } from '@/components/Icon'

export function QuickLinks({ children }) {
  return (
    <div className="not-prose my-12 grid grid-cols-1 gap-6 sm:grid-cols-2">
      {children}
    </div>
  )
}

export function QuickLink({ title, description, href, icon }) {
  // COMMENT: 4 WINDOWS ON THE MAIN SCREEN
  return (
    <div className="group relative rounded-xl border border-zinc-200 dark:border-zinc-800">
      <div className="absolute -inset-px rounded-xl border-2 border-transparent opacity-0 [background:linear-gradient(var(--quick-links-hover-bg,theme(colors.orange.50)),var(--quick-links-hover-bg,theme(colors.orange.50)))_padding-box,linear-gradient(to_top,theme(colors.red.500),theme(colors.orange.500),theme(colors.orange.400))_border-box] group-hover:opacity-100 dark:[--quick-links-hover-bg:theme(colors.zinc.800)]" />
      <div className="relative overflow-hidden rounded-xl p-6">
        <Icon icon={icon} className="h-8 w-8" />
        <h2 className="mt-4 font-display text-base text-zinc-900 dark:text-white">
          <Link href={href}>
            <span className="absolute -inset-px rounded-xl" />
            {title}
          </Link>
        </h2>
        <p className="mt-1 text-sm text-zinc-700 dark:text-zinc-400">
          {description}
        </p>
      </div>
    </div>
  )
}
