#import "@preview/cetz:0.3.1": canvas, draw, coordinate, util, intersection, vector

#let report(
  title: none,
  course: none,
  authors: (),
  university: none,
  reference: none,
  bibliography-path: "",
  nb-columns: 1,
  abstract: none,
  doc
) = {
  set text(size: 12pt, lang: "en", font: "New Computer Modern")

  show math.equation: set block(breakable: true)

  set enum(numbering: "1. a.")
  set list(marker: [--])

  set page(
    numbering: "1",
    margin: (x: 2cm, y: 3cm),
  )

  set par(justify: true, spacing: 1.475em)

  pad(x: 1.5cm, align(center)[
    #v(.5cm)
    #rect(width: 100%, inset: .4cm, stroke: (top: 1.6pt, bottom: .8pt))[
      = #title
    ]
    #v(.5cm)
  ])

  pad(x: 1.6cm, align(center)[
    #set text(size: .95em)
    #grid(
      columns: (1fr,) * calc.min(authors.len(), 2),
      row-gutter: 2em,
      ..authors.map(author => [
        #text(weight: 700, author.name) \
        #text(author.affiliation) \
        #text(raw(author.email))
      ]),
    )
  ])

  v(1.5em)

  if abstract != none {
    align(center, text(size: 1.1em, weight: 700, "Abstract"))
    pad(x: 2cm, bottom: 0.5cm, abstract)
  }

  v(1em)

  show heading.where(
    level: 2
  ): it => block(width: 100%)[
    #set align(center)
    #set text(1.0em, weight: 700)
    #pad(x: 1cm, smallcaps(it.body))
    #v(0.45cm)
  ]

  show heading.where(
    level: 3
  ): it => text(size: 1em, weight: 700, style: "italic", it.body + [.])

  show heading.where(
    level: 4
  ): it => text(size: 1em, weight: 700, style: "italic", h(1em) + [(] + it.body + [)])

  if nb-columns > 1 {
    show: rest => columns(nb-columns, rest)
    doc

    if bibliography-path != "" {
      bibliography(title: [ == Bibliographie ], bibliography-path, style: "association-for-computing-machinery")
    }
  } else {
    doc

    if bibliography-path != "" {
      bibliography(title: [ == Bibliographie ], bibliography-path, style: "association-for-computing-machinery")
    }
  }
}

#let hidden-bib(body) = {
  box(width: 0pt, height: 0pt, hide(body))
}

#let bar(value) = math.accent(value, "-")
#let argmin = { "argmin" }
#let argmax = { "argmax" }

#let fig(
  title: none,
  description: none,
  path: "",
  width: auto,
  height: auto,
  link: "",
) = [
  #v(1.15em)
  #figure(
    caption: [ *#title.* #description ],
    image(width: width, height: height, path)
  ) #label(link)
  #v(1.15em)
]
