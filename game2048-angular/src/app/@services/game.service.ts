/**
 * author: Foromo Daniel Soromou
 */

 import { Injectable }  from '@angular/core';
 import * as from2048 from '../@stores';
 import * as GameActions from '../@stores/game-action';
 import { select, Store } from '@ngrx/store';
 import { Tile } from '../@stores/tile.model';
 

 export const SIZE = 4;
 export const STARTING_TILES = 2;

 const Vectors = {
     37: {x: -1, y: 0}, // left
     38: {x: 0, y: -1}, // up
     39: {x: 1, y: 0}, // right
     40: {x: 0, y: 1} // down
 };

 @Injectable() 
 export class GameService {
    private _cells: string[];
    get cells(): string[] {
        return this._cells;
    }

    private _tiles: Tile[];

    private _gameStats: any;

    constructor( private store: Store<from2048.State> ) {
        this.store.pipe(select(from2048.getTiles)).subscribe(( tiles ) => {
            this._tiles = tiles;
        });

        this.store.pipe(select(from2048.getGameStats)).subscribe(( stats: any ) => {
            this._gameStats = stats;
        });
    }

    public newGame(): void {
        this.resetGameStatus();
        this.buildGameGrid();
        this.buildStartingPosition();
    }

    public move( direction: number ): void {

        if (this._gameStats.gameOver) {
            return;
        }

        let hasMoved = false;
        let scores = 0;
        let highestScores = this._gameStats.highestScores; // a variable to hold the highest scores
        let gameWon = false;
        let gameOver = false;

        this.ResetTilesStatus();

        const positions = this.getTraversalDirections(direction);

        positions.x.forEach(( x ) => {
            positions.y.forEach(( y ) => {
                const originalCoordinate = {x, y};
                const originalPosition = x + y * SIZE;
                const tile = this.getTileAt(originalCoordinate);

                if (tile) {
                    const cell = this.calculateNextPosition(originalCoordinate, direction);
                    const next = cell.next;

                    if (next && next.value === tile.value && !next.merged) {
                        next.value = tile.value * 2; 

                        tile.shouldDump = true;
                        next.merged = true;

                        this.moveTile(next.position, tile, next);

                        scores += next.value;

                        hasMoved = true;

                        if (next.value >= 2048 && !gameWon) {
                            gameWon = true;
                        }

                    } else {
                        this.moveTile(cell.newPosition, tile);
                        this._cells[cell.newPosition] = tile.id;
                    }

                    if (!hasMoved && originalPosition !== cell.newPosition) {
                        hasMoved = true;
                    }
                }
            });
        });

        // Update the game score after grid checking
        if (scores > 0) {
            scores = this._gameStats.scores + scores;
            if (scores > highestScores) {
                highestScores = scores;
                localStorage.setItem('2048-best', highestScores.toString());
            }
        } else {
            scores = this._gameStats.scores;
        }

        if (hasMoved) {
            this.insertTileRandomly();

            gameOver = !this.anyMoveAvailable();
        }

        this.store.dispatch(new GameActions.SetGameStats({
            gameOver, gameWon, scores, highestScores
        }));
    }

    private resetGameStatus() {
        this.store.dispatch(new GameActions.ResetGame());
    }

    private buildGameGrid(): void {
        this._cells = new Array(SIZE * SIZE).fill(null);
    }

    private buildStartingPosition(): void {
        for (let i = 0; i < STARTING_TILES; i++) {
            this.insertTileRandomly();
        }
    }

    private insertTileRandomly(): void {
        const cells = this.getAllAvailableCells();

        if (cells && cells.length > 0) {
            const randomCell = cells[Math.floor(Math.random() * cells.length)];
            const newTile = new Tile(randomCell);

            this._cells[randomCell] = newTile.id;

            this.store.dispatch(new GameActions.AddTile(newTile));
        }
    }

    private getAllAvailableCells(): number[] {
        return this._cells.map(( cell: string, index: number ) => {
            return cell === null ? index : null;
        }).filter(( value: number ) => value !== null);
    }

    private getTraversalDirections( direction: number ): { x: number[], y: number[] } {
        const vector = Vectors[direction];
        const positions: { x: number[], y: number[] } = {x: [], y: []};
        for (let i = 0; i < SIZE; i++) {
            positions.x.push(i);
            positions.y.push(i);
        }

        if (vector.y > 0) {
            positions.y = positions.y.reverse();
        }

        if (vector.x > 0) {
            positions.x = positions.x.reverse();
        }

        return positions;
    }

    private getTileAt( coordination: { x: number, y: number } ): Tile {

        if (this.withinGrid(coordination)) {
            const position = coordination.x + coordination.y * SIZE;
            const id = this._cells[position];

            // Once we get the tile's id from grid,
            // we can get the specific tile from tiles array
            return this._tiles.find(( t ) => t.id === id);
        } else {
            return null;
        }
    }

    /* Calculate a next possible coordination a tile could move to */
    private calculateNextPosition( coordination: { x: number, y: number }, direction: number ): any {
        // the move vector based on the move direction
        const vector = Vectors[direction];
        let current: { x: number, y: number };
        do {
            // Save the current coordination
            current = coordination;

            // Get the next coordination
            coordination = {
                x: current.x + vector.x,
                y: current.y + vector.y
            };
        } while (this.checkCellAvailability(coordination));

        const newPosition = current.x + current.y * SIZE;

        return {
            newPosition, 
            next: this.getTileAt(coordination) 
        };
    }

    private checkCellAvailability( coordination: { x: number, y: number } ): boolean {
        if (this.withinGrid(coordination)) {
            const index = coordination.x + coordination.y * SIZE;
            return !this._cells[index];
        } else {
            return false;
        }
    }

    private moveTile( newPosition: number, oldTile: Tile, nextTile: Tile = null ): void {
        const oldPosition = oldTile.position;
        if (newPosition === oldPosition) {
            return;
        }

        this._cells[oldPosition] = null;

        this.store.dispatch(new GameActions.UpdateTiles({newPosition, oldTile, nextTile}));
    }

    private ResetTilesStatus(): void {
        this.store.dispatch(new GameActions.ResetTileStatus());
    }

    private anyMoveAvailable(): boolean {
        return this.getAllAvailableCells().length > 0 || this.tileMatchesAvailable();
    }

    /* Check if any two joint tiles could be merged */
    public tileMatchesAvailable(): boolean {
        for (let i = 0; i < SIZE * SIZE; i++) {
            const coordination = {x: i % SIZE, y: Math.floor(i / SIZE)};
            const tile = this.getTileAt(coordination);

            if (tile) {
                for (const vectorName in Vectors) {
                    if (vectorName) {
                        const vector = Vectors[vectorName];
                        const otherCoordi = {x: coordination.x + vector.x, y: coordination.y + vector.y};
                        const other = this.getTileAt(otherCoordi);

                        if (other && other.value === tile.value) {
                            return true;
                        }
                    }
                }
            }
        }

        return false;
    }

    private withinGrid( coordination: { x: number, y: number } ): boolean {
        return coordination.x >= 0 && coordination.x < SIZE &&
            coordination.y >= 0 && coordination.y < SIZE;
    }
 }