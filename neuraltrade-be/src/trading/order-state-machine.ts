import { Logger } from '@nestjs/common';
import { EventEmitter2 } from '@nestjs/event-emitter';

/**
 * Order State Machine
 * 
 * Manages order lifecycle transitions:
 * PENDING → SUBMITTED → OPEN → [FILLED | PARTIAL | CANCELLED | REJECTED]
 * 
 * Ensures valid state transitions and emits events on state changes.
 */

// ==========================================
// Order States
// ==========================================

export type OrderState =
    | 'PENDING'     // Order created, not yet sent to exchange
    | 'SUBMITTED'   // Sent to exchange, awaiting confirmation
    | 'OPEN'        // Confirmed by exchange, waiting to fill
    | 'PARTIAL'     // Partially filled
    | 'FILLED'      // Completely filled
    | 'CANCELLED'   // Cancelled by user or system
    | 'REJECTED'    // Rejected by exchange
    | 'EXPIRED';    // Expired (for GTD orders)

// ==========================================
// State Transition Events
// ==========================================

export type OrderEvent =
    | 'SUBMIT'      // Submit to exchange
    | 'CONFIRM'     // Exchange confirmed
    | 'PARTIAL_FILL'// Partial fill received
    | 'FILL'        // Complete fill received
    | 'CANCEL'      // Cancel requested
    | 'REJECT'      // Exchange rejected
    | 'EXPIRE';     // Order expired

// ==========================================
// Valid Transitions
// ==========================================

const STATE_TRANSITIONS: Record<OrderState, Partial<Record<OrderEvent, OrderState>>> = {
    PENDING: {
        SUBMIT: 'SUBMITTED',
        CANCEL: 'CANCELLED',
    },
    SUBMITTED: {
        CONFIRM: 'OPEN',
        REJECT: 'REJECTED',
        FILL: 'FILLED',         // Market orders can fill immediately
        PARTIAL_FILL: 'PARTIAL',
    },
    OPEN: {
        PARTIAL_FILL: 'PARTIAL',
        FILL: 'FILLED',
        CANCEL: 'CANCELLED',
        EXPIRE: 'EXPIRED',
    },
    PARTIAL: {
        PARTIAL_FILL: 'PARTIAL',
        FILL: 'FILLED',
        CANCEL: 'CANCELLED',
    },
    FILLED: {},     // Terminal state
    CANCELLED: {},  // Terminal state
    REJECTED: {},   // Terminal state
    EXPIRED: {},    // Terminal state
};

// ==========================================
// Order State Machine Class
// ==========================================

export class OrderStateMachine {
    private readonly logger = new Logger(OrderStateMachine.name);

    constructor(
        private readonly orderId: number,
        private currentState: OrderState,
        private readonly eventEmitter?: EventEmitter2,
    ) { }

    /**
     * Get current state
     */
    getState(): OrderState {
        return this.currentState;
    }

    /**
     * Check if transition is valid
     */
    canTransition(event: OrderEvent): boolean {
        const transitions = STATE_TRANSITIONS[this.currentState];
        return event in transitions;
    }

    /**
     * Get next state for an event (without transitioning)
     */
    getNextState(event: OrderEvent): OrderState | null {
        const transitions = STATE_TRANSITIONS[this.currentState];
        return transitions[event] || null;
    }

    /**
     * Transition to next state
     * Returns true if transition was successful
     */
    transition(event: OrderEvent): boolean {
        const nextState = this.getNextState(event);

        if (!nextState) {
            this.logger.warn(
                `Invalid transition: Order ${this.orderId} cannot transition from ${this.currentState} via ${event}`
            );
            return false;
        }

        const previousState = this.currentState;
        this.currentState = nextState;

        this.logger.debug(
            `Order ${this.orderId}: ${previousState} → ${nextState} (${event})`
        );

        // Emit event for listeners
        if (this.eventEmitter) {
            this.eventEmitter.emit('order.state.changed', {
                orderId: this.orderId,
                previousState,
                currentState: this.currentState,
                event,
                timestamp: new Date(),
            });
        }

        return true;
    }

    /**
     * Check if order is in terminal state
     */
    isTerminal(): boolean {
        return ['FILLED', 'CANCELLED', 'REJECTED', 'EXPIRED'].includes(this.currentState);
    }

    /**
     * Check if order is active (can be modified/cancelled)
     */
    isActive(): boolean {
        return ['PENDING', 'SUBMITTED', 'OPEN', 'PARTIAL'].includes(this.currentState);
    }

    /**
     * Check if order is waiting for fill
     */
    isAwaitingFill(): boolean {
        return ['SUBMITTED', 'OPEN', 'PARTIAL'].includes(this.currentState);
    }
}

// ==========================================
// Factory Function
// ==========================================

export function createOrderStateMachine(
    orderId: number,
    initialState: OrderState = 'PENDING',
    eventEmitter?: EventEmitter2,
): OrderStateMachine {
    return new OrderStateMachine(orderId, initialState, eventEmitter);
}

// ==========================================
// State Change Event Type
// ==========================================

export interface OrderStateChangedEvent {
    orderId: number;
    previousState: OrderState;
    currentState: OrderState;
    event: OrderEvent;
    timestamp: Date;
}
