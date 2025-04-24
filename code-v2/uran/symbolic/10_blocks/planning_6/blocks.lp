#include <incmode>.
#const imax = 51.

block(a;b;c;d;e;f).

init(on(a,e)).
init(on(b,c)).
init(on(c,table)).
init(on(d,f)).
init(on(e,table)).
init(on(f,table)).

goal(on(a,table)).
goal(on(b,a)).
goal(on(c,b)).
goal(on(d,c)).
goal(on(e,d)).
goal(on(f,e)).

holds(F,0) :- init(F).

#program step(t).

{move(X,Y,t) : block(X), block(Y), X != Y;
 move(X,table,t) : block(X)} = 1.

:- move(X,Y,t), holds(on(X,Y),t-1).
:- move(X,Y,t), holds(on(Z,X),t-1).
:- move(X,Y,t), holds(on(Z,Y),t-1), block(Y).

holds(on(X,Y),t) :- move(X,Y,t), block(X).

moved(X,t) :- move(X,Y,t), block(X).
holds(on(X,Y),t) :- holds(on(X,Y),t-1), not moved(X,t), block(X).

#program check(t).

:- query(t), goal(F), not holds(F,t).

#show move/3.
