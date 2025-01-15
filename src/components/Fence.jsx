'use client'

import { Fragment } from 'react'
import { Highlight } from 'prism-react-renderer'

export function Fence({ children, language }) {
  return (
    <Highlight
      code={children.trimEnd()}
      language={language}
      theme={{plain: {color:'#e4e4e7', fontStyle:'normal',}, // zinc-200
      styles: [
        {
          types: ['builtin', 'changed', 'keyword', 'function'],
          style: {
            color: '#ff7500' // orange
          }
        },
        {
          types: ['string', 'number', 'definition', 'boolean', 'class-name'],
          style: {
            color: '#fdba74' // orange-300
          }
        },
        {
          types: ['comment'],
          style: {
            color: '#6a9955'// green
          }
        },
        {
          types: ['property', 'punctuation', 'operator'],
          style: {
            color: '#a1a1aa' // zinc-400
          }
        },
      ]
      }} //https://github.com/PrismJS/prism/blob/master/components/prism-al.js
    >
      {({ className, style, tokens, getTokenProps }) => (
        <pre className={className} style={style}>
          <code>
            {tokens.map((line, lineIndex) => (
              <Fragment key={lineIndex}>
                {line
                  .filter((token) => !token.empty)
                  .map((token, tokenIndex) => (
                    <span key={tokenIndex} {...getTokenProps({ token })} />
                  ))}
                {'\n'}
              </Fragment>
            ))}
          </code>
        </pre>
      )}
    </Highlight>
  )
}