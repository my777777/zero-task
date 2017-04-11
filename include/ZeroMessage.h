/*
 * ZeroMessage.h
 *
 *  Created on: Dec 19, 2016
 *      Author: zys
 */

#ifndef ZEROMESSAGE_H_
#define ZEROMESSAGE_H_

enum MESSAGES {

	NET_CAN_FPROP, NET_CAN_BPROP, NET_CAN_UPDATE, NET_CAN_SYNC, NET_CAN_EXIT, NET_CAN_TEST,NET_END_TEST,NET_CAN_CALACC,

	LAYER_CAN_BPROP, LAYER_CAN_FPROP, LAYER_CAN_UPDATE, LAYER_CAN_SYNC, LAYER_CAN_EXIT
};
class ZeroMessage {

protected:
	MESSAGES _messageType;
public:
	MESSAGES getType() {
		return _messageType;
	}
	virtual ZeroMessage* clone() {
		return new ZeroMessage(_messageType);
	}
	ZeroMessage(MESSAGES messageType) :
			_messageType(messageType) {
	}
	virtual ~ZeroMessage() {
	}
};

class PropMessage: public ZeroMessage {

protected:
	int _toLayerId;
	int _fragment;
public:

	int getToLayerId() {
		return _toLayerId;
	}

	int getFragment(){
		return _fragment;
	}

	virtual PropMessage* clone() {
		return new PropMessage(_toLayerId, _fragment, _messageType);
	}

	PropMessage(int toLayerId, int fragment, MESSAGES msgType) :
			ZeroMessage(msgType), _toLayerId(toLayerId), _fragment(fragment) {
	}
};

class FpropMessage: public PropMessage {
public:
	FpropMessage(int toLayerId) :
			PropMessage(toLayerId, 0, LAYER_CAN_FPROP) {
	}
	FpropMessage(int toLayerId, int fragment) :
			PropMessage(toLayerId, fragment, LAYER_CAN_FPROP) {
	}
	virtual FpropMessage* clone() {
		return new FpropMessage(_toLayerId);
	}
};

class BpropMessage: public PropMessage {
public:
	BpropMessage(int toLayerId) :
			PropMessage(toLayerId, 0, LAYER_CAN_BPROP) {
	}

	BpropMessage(int toLayerId, int fragment) :
			PropMessage(toLayerId, fragment, LAYER_CAN_BPROP) {
	}
	virtual BpropMessage* clone() {
		return new BpropMessage(_toLayerId);
	}
};
class UpdateMessage: public PropMessage {
public:
	UpdateMessage(int toLayerId) :
			PropMessage(toLayerId, 0,LAYER_CAN_UPDATE) {
	}
	UpdateMessage(int toLayerId, int fragment) :
			PropMessage(toLayerId, fragment, LAYER_CAN_UPDATE) {
	}
	virtual UpdateMessage* clone() {
		return new UpdateMessage(_toLayerId);
	}
};
#endif /* ZEROMESSAGE_H_ */
