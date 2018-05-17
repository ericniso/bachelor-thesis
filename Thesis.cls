%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Thesis LaTeX Template - THESIS CLASS FILE
%
% The original template was downloaded from:
% http://www.latextemplates.com
%
% The current version of the class file borrows heavily from the one available
% on latextemplates.com but has been modified to meet the needs of those trying
% to create a report for presentation in BITS Pilani.
%
% This class file defines the structure and design of the template.
%
% There is one part that needs to be filled out - the variables
% dictating the document particulars such as the author name, university
% name, etc. You will find these in the variables.tex file.
%
% The other two easily-editable sections are the margin sizes and abstract.
% These have both been commented for easy editing. Advanced LaTeX
% users will have no trouble editing the rest of the document to their liking.
%
% Original header:
%% This is file `Thesis.cls', based on 'ECSthesis.cls', by Steve R. Gunn
%% generated with the docstrip utility.
%%
%% Created by Steve R. Gunn, modified by Sunil Patel: www.sunilpatel.co.uk
%% Further modified by www.latextemplates.com. Later modified by Darshit Shah
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%-------------------------------------------------------------------------------
% Base Class Definition
%-------------------------------------------------------------------------------
\NeedsTeXFormat{LaTeX2e}[1996/12/01]
\ProvidesClass{Thesis}
              [2007/22/02 v1.0
   LaTeX document class]
\def\baseclass{book}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{\baseclass}}
\def\@checkoptions#1#2{
  \edef\@curroptions{\@ptionlist{\@currname.\@currext}}
  \@tempswafalse
  \@tfor\@this:=#2\do{
    \@expandtwoargs\in@{,\@this,}{,\@curroptions,}
    \ifin@ \@tempswatrue \@break@tfor \fi}
  \let\@this\@empty
  \if@tempswa \else \PassOptionsToClass{#1}{\baseclass}\fi
}
\@checkoptions{11pt}{{10pt}{11pt}{12pt}}
\PassOptionsToClass{a4paper}{\baseclass}
\ProcessOptions\relax
\LoadClass{\baseclass}
%-------------------------------------------------------------------------------

\RequirePackage[utf8]{inputenc} % Allows the use of international characters (e.g. Umlauts)

\newcommand\bhrule{\typeout{------------------------------------------------------------------------------}}
\newcommand\btypeout[1]{\bhrule\typeout{\space #1}\bhrule}
\newcommand{\HRule}{\rule{\linewidth}{0.5mm}} % New command to make the lines in the title page

\def\today{\ifcase\month\or
  January\or February\or March\or April\or May\or June\or
  July\or August\or September\or October\or November\or December\fi
  \space \number\year}

%-------------------------------------------------------------------------------
% LANGUAGES
%-------------------------------------------------------------------------------
\usepackage[english,italian]{babel}

%-------------------------------------------------------------------------------
% FIRST PAGE INCLUDE
%-------------------------------------------------------------------------------
\usepackage{pdfpages}

%-------------------------------------------------------------------------------
% REMOVE CHAPTER NUMBER
%-------------------------------------------------------------------------------
\usepackage{titlesec}
\titleformat{\chapter}[display]
    {\normalfont\bfseries}{}{0pt}{\Huge}

%-------------------------------------------------------------------------------
% SPACING RULES
%-------------------------------------------------------------------------------
\usepackage{setspace}
\onehalfspacing
\setlength{\parindent}{0pt}
\setlength{\parskip}{2.0ex plus0.5ex minus0.2ex}
%-------------------------------------------------------------------------------

%-------------------------------------------------------------------------------
% MARGINS
%-------------------------------------------------------------------------------
\usepackage{vmargin}
\setmarginsrb  { 1.0in}  % left margin
               { 0.6in}  % top margin
               { 1.0in}  % right margin
               { 0.8in}  % bottom margin
               {  20pt}  % head height
               {0.25in}  % head sep
               {   9pt}  % foot height
               { 0.3in}  % foot sep
%-------------------------------------------------------------------------------

%-------------------------------------------------------------------------------
% QUOTATION PAGE
%
% Use the command \Quotation{Quote}{Author} to create a single page with a
% quotation in the document.
%-------------------------------------------------------------------------------
\newcommand\Quotation[2]{
    \btypeout{Quotation}
    \pagestyle{empty} % No headers or footers for the following pages

    \null\vfill % Add some space to move the quote down the page a bit

    \textit{``#1''}

    \begin{flushright}
        #2
    \end{flushright}

    % Add some space at the bottom to position the quote just right
    \vfill\vfill\vfill\vfill\vfill\vfill\null

    \clearpage % Start a new page
}
%-------------------------------------------------------------------------------

\raggedbottom
\setlength{\topskip}{1\topskip \@plus 5\p@}
\doublehyphendemerits=10000       % No consecutive line hyphens.
\brokenpenalty=10000              % No broken words across columns/pages.
\widowpenalty=9999                % Almost no widows at bottom of page.
\clubpenalty=9999                 % Almost no orphans at top of page.
\interfootnotelinepenalty=9999    % Almost never break footnotes.
\usepackage{fancyhdr}
\lhead[\rm\thepage]{\fancyplain{}{\sl{\rightmark}}}
\rhead[\fancyplain{}{\sl{\leftmark}}]{\rm\thepage}
\chead{}\lfoot{}\rfoot{}\cfoot{}
\pagestyle{fancy}
\renewcommand{\chaptermark}[1]{\btypeout{\thechapter\space #1}\markboth{\@chapapp\ \thechapter\ #1}{\@chapapp\ \thechapter\ #1}}
\renewcommand{\sectionmark}[1]{}
\renewcommand{\subsectionmark}[1]{}
\def\cleardoublepage{\clearpage\if@twoside \ifodd\c@page\else
\hbox{}
\thispagestyle{empty}
\newpage
\if@twocolumn\hbox{}\newpage\fi\fi\fi}
\usepackage{amsmath,amsfonts,amssymb,amscd,amsthm,xspace,mathtools}
\theoremstyle{plain}
\newtheorem{example}{Example}[chapter]
\newtheorem{theorem}{Theorem}[chapter]
\newtheorem{corollary}[theorem]{Corollary}
\newtheorem{lemma}[theorem]{Lemma}
\newtheorem{proposition}[theorem]{Proposition}
\newtheorem{axiom}[theorem]{Axiom}
\theoremstyle{definition}
\newtheorem{definition}[theorem]{Definition}
\theoremstyle{remark}
\newtheorem{remark}[theorem]{Remark}
\usepackage[centerlast,small,sc]{caption}
\setlength{\captionmargin}{20pt}
\newcommand{\fref}[1]{Figure~\ref{#1}}
\newcommand{\tref}[1]{Table~\ref{#1}}
\newcommand{\eref}[1]{Equation~\ref{#1}}
\newcommand{\cref}[1]{Chapter~\ref{#1}}
\newcommand{\sref}[1]{Section~\ref{#1}}
\newcommand{\aref}[1]{Appendix~\ref{#1}}
\renewcommand{\topfraction}{0.85}
\renewcommand{\bottomfraction}{.85}
\renewcommand{\textfraction}{0.1}
\renewcommand{\dbltopfraction}{.85}
\renewcommand{\floatpagefraction}{0.75}
\renewcommand{\dblfloatpagefraction}{.75}
\setcounter{topnumber}{9}
\setcounter{bottomnumber}{9}
\setcounter{totalnumber}{20}
\setcounter{dbltopnumber}{9}
\usepackage{graphicx}
\usepackage{epstopdf}
\usepackage{booktabs}
\usepackage{rotating}
\usepackage{enumitem}
\usepackage{listings}
\usepackage{lstpatch}
\usepackage{microtype}
\lstset{captionpos=b,
        frame=tb,
        basicstyle=\scriptsize\ttfamily,
        showstringspaces=false,
        keepspaces=true}
\lstdefinestyle{matlab} {
        language=Matlab,
        keywordstyle=\color{blue},
        commentstyle=\color[rgb]{0.13,0.55,0.13}\em,
        stringstyle=\color[rgb]{0.7,0,0} }
\usepackage[pdfpagemode={UseOutlines},bookmarks=true,bookmarksopen=true,
   bookmarksopenlevel=0,bookmarksnumbered=true,hypertexnames=false,
   colorlinks,linkcolor={black},citecolor={black},urlcolor={black},
   pdfstartview={FitV},unicode,breaklinks=true]{hyperref}

\usepackage[toc]{glossaries}
\glossarystyle{long}
\makeglossaries
\DeclarePairedDelimiter\ceil{\lceil}{\rceil}
\DeclarePairedDelimiter\floor{\lfloor}{\rfloor}
\pdfstringdefDisableCommands{
   \let\\\space
}

\newcommand*{\thesistitle}[1]{\def\ttitle{#1}}
\newcommand*{\supervisor}[1]{\def\supname{#1}}
\newcommand*{\cosupervisor}[1]{\def\cosupname{#1}}
\newcommand*{\documenttype}[1]{\def\doctype{#1}}
\newcommand*{\coursecode}[1]{\def\ccode{#1}}
\newcommand*{\coursename}[1]{\def\cname{#1}}
\newcommand*{\examiner}[1]{\def\examname{#1}}
\newcommand*{\degree}[1]{\def\degreename{#1}}
\newcommand*{\authors}[1]{\def\authornames{#1}}
\newcommand*{\IDNumber}[1]{\def\idnum{#1}}
\newcommand*{\addresses}[1]{\def\addressnames{#1}}
\newcommand*{\university}[1]{\def\univname{#1}}
\newcommand*{\UNIVERSITY}[1]{\def\UNIVNAME{#1}}
\newcommand*{\department}[1]{\def\deptname{#1}}
\newcommand*{\DEPARTMENT}[1]{\def\DEPTNAME{#1}}
\newcommand*{\group}[1]{\def\groupname{#1}}
\newcommand*{\GROUP}[1]{\def\GROUPNAME{#1}}
\newcommand*{\faculty}[1]{\def\facname{#1}}
\newcommand*{\FACULTY}[1]{\def\FACNAME{#1}}
\newcommand*{\supervisorposition}[1]{\def\suppos{#1}}
\newcommand*{\supervisorinstitute}[1]{\def\supinst{#1}}
\newcommand*{\cosupervisorposition}[1]{\def\cosuppos{#1}}
\newcommand*{\cosupervisorinstitute}[1]{\def\cosupinst{#1}}
\newcommand*{\subject}[1]{\def\subjectname{#1}}
\newcommand*{\keywords}[1]{\def\keywordnames{#1}}

%-------------------------------------------------------------------------------
% ABSTRACT PAGE DESIGN
%-------------------------------------------------------------------------------
\newenvironment{abstract}
{
    \btypeout{Abstract}
    \addtotoc{Abstract}
    \addtocontents{toc}{\vspace{1em}}
    \setstretch{1.3}
    \thispagestyle{plain}
    \begin{center}{\huge{\textit{Abstract}} \par}\end{center}
    \normalsize
}
{
    \vfil\vfil\null
    \clearpage
}
%-------------------------------------------------------------------------------

%-------------------------------------------------------------------------------
% ACKNOWLEDGEMENTS
%-------------------------------------------------------------------------------
\newenvironment{acknowledgements}
{
    \btypeout{Riconoscimenti}
    \addtotoc{Riconoscimenti}
    \addtocontents{toc}{\vspace{1em}}
    \setstretch{1.3}
    \thispagestyle{plain}
    \begin{center}{\huge{\textit{Riconoscimenti}} \par}\end{center}
    \normalsize
}
{
    \vfil\vfil\null
    \clearpage
}
%-------------------------------------------------------------------------------

%-------------------------------------------------------------------------------
% DEDICATORY
%-------------------------------------------------------------------------------
\newcommand\Dedicatory[1]{
\btypeout{Dedicatory}
\thispagestyle{plain}
\null\vfil
\vskip 60\p@
\begin{center}{\Large \sl #1}\end{center}
\vfil\null
\cleardoublepage
}
%-------------------------------------------------------------------------------

\addtocounter{secnumdepth}{1}
\setcounter{tocdepth}{3}
\newcounter{dummy}
\newcommand\addtotoc[1]{
\refstepcounter{dummy}
\addcontentsline{toc}{chapter}{#1}}
\renewcommand\tableofcontents{
\btypeout{Tabella dei Contenuti}
%\addtotoc{Contenuti}
\begin{spacing}{1}{
    \setlength{\parskip}{1pt}
    \if@twocolumn
      \@restonecoltrue\onecolumn
    \else
      \@restonecolfalse
    \fi
    \chapter*{\contentsname
        \@mkboth{
           \MakeUppercase\contentsname}{\MakeUppercase\contentsname}}
    \@starttoc{toc}
    \if@restonecol\twocolumn\fi
   \cleardoublepage
}\end{spacing}
}
\renewcommand\listoffigures{
\btypeout{Elenco delle Figure}
\addtotoc{Elenco delle Figure}
\begin{spacing}{1}{
    \setlength{\parskip}{1pt}
    \if@twocolumn
      \@restonecoltrue\onecolumn
    \else
      \@restonecolfalse
    \fi
    \chapter*{\listfigurename
      \@mkboth{\MakeUppercase\listfigurename}
              {\MakeUppercase\listfigurename}}
    \@starttoc{lof}
    \if@restonecol\twocolumn\fi
    \cleardoublepage
}\end{spacing}
}
\renewcommand\listoftables{
\btypeout{Elenco delle Tabelle}
\addtotoc{Elenco delle Tabelle}
\begin{spacing}{1}{
    \setlength{\parskip}{1pt}
    \if@twocolumn
      \@restonecoltrue\onecolumn
    \else
      \@restonecolfalse
    \fi
    \chapter*{\listtablename
      \@mkboth{
          \MakeUppercase\listtablename}{\MakeUppercase\listtablename}}
    \@starttoc{lot}
    \if@restonecol\twocolumn\fi
    \cleardoublepage
}\end{spacing}
}
\renewcommand\backmatter{
  \if@openright
    \cleardoublepage
  \else
    \clearpage
  \fi
  \addtotoc{\bibname}
  \btypeout{\bibname}
  \@mainmatterfalse}
\endinput
