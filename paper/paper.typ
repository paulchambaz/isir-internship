#import "template.typ": *
#import "@preview/cetz:0.3.1": canvas, draw

#show: report.with(
  title: [ Article title ],
  authors: (
    (
      name: "Paul Chambaz",
      affiliation: "Sorbonne Université",
      email: "paul.chambaz@tutanota.com",
    ),
  ),
  nb-columns: 2,
  abstract: [
    #lorem(40)
  ]
)

== Header

=== Subheader
#lorem(50)

#lorem(150)

=== Subheader
#lorem(350)

#fig(
  title: [Title],
  description: [#lorem(20)],
  path: "./figures/sin.svg",
  link: "fig-1"
)

#lorem(500)
