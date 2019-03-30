/**
 * author: Foromo Daniel Soromou
 */

import * as fromGame from './game-reducer';
import { ActionReducerMap, createFeatureSelector, createSelector } from '@ngrx/store';
import { dependenciesFromGlobalMetadata } from '@angular/compiler/src/render3/r3_factory';


export interface State {
    game: fromGame.State;
}

export const reducers: ActionReducerMap<State> = {
    game: fromGame.reducer,
}

export const getGameState = createFeatureSelector('game')

export const getTiles = createSelector(getGameState, fromGame.getTiles);

export const getGameStats = createSelector(getGameState, fromGame.getGameStats);