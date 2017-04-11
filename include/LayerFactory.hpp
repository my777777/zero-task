/*
 * LayerFactory.h
 *
 *  Created on: Mar 1, 2017
 *      Author: zys
 */

#ifndef LAYERFACTORY_H_
#define LAYERFACTORY_H_

#include "Resource.h"
#include "Layer.h"

class LayerFactory {
private:
	LayerFactory(){};

public:
	typedef Layer* (*Creator)(const LayerParam&);
	typedef map<string, Creator> CreatorRegistry;

	static CreatorRegistry& Registry() {
		static CreatorRegistry* g_registry = new CreatorRegistry();
		return *g_registry;
	}

	// Adds a creator.
	static void AddCreator(const string& type, Creator creator) {
		CreatorRegistry& registry = Registry();
		CHECK_EQ(registry.count(type), 0)<< "Layer type " << type << " already registered.";
		registry[type] = creator;
	}

	// Get a layer using a LayerParam.
	static Layer* CreateLayer(const LayerParam& param) {
		const string& type = param.getType();
		CreatorRegistry& registry = Registry();
		CHECK_EQ(registry.count(type), 1) << "Unknown layer type: " << type;
		return registry[type](param);
	}
	virtual ~LayerFactory(){};
};

class LayerRegisterer {
 public:
  LayerRegisterer(const string& type,Layer* (*creator)(const LayerParam&)) {
    // LOG(INFO) << "Registering layer type: " << type;
    LayerFactory::AddCreator(type, creator);
  }
};

#define REGISTER_LAYER_CREATOR(type, creator)  								\
	static LayerRegisterer g_creator_d_##type(#type, creator)				\

#define REGISTER_LAYER_CLASS(type)                                          \
  Layer* Creator_##type##Layer(const LayerParam& param)                 	\
  {                                                                         \
    return (Layer*)(new type##Layer(param));           						\
  }                                                                         \
  REGISTER_LAYER_CREATOR(type, Creator_##type##Layer)

#endif /* LAYERFACTORY_H_ */











