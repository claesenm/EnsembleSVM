//  (C) Copyright Thomas Becker 2005. Permission to copy, use, modify, sell and
//  distribute this software is granted provided this copyright notice appears
//  in all copies. This software is provided "as is" without express or implied
//  warranty, and with no claim as to its suitability for any purpose.

// Revision History
// ================
//
// 27 Dec 2006 (Thomas Becker) Created

#ifndef ANY_ITERATOR_ABSTRACT_BASE_01102007TMB_HPP
#define ANY_ITERATOR_ABSTRACT_BASE_01102007TMB_HPP

// Includes
// ========

#include "any_iterator_metafunctions.hpp"
#include <boost/iterator/iterator_categories.hpp>
#include <boost/type_traits/add_const.hpp>
#include <boost/type_traits/remove_const.hpp>

namespace IteratorTypeErasure
{

  namespace detail
  {

    ///////////////////////////////////////////////////////////////////////
    // 
    // The partial specializations of any_iterator_abstract_base (which is
    // the equivalent of boost::any::placeholder) mirror the hierarchy of
    // boost's iterator traversal tags.
    //
    // The first four template arguments are as in boost::iterator_facade.
    // The last template argument is the traversal tag of the most
    // derived class of the current instantiation of the hierarchy. This
    // is a slight variant of the CRTP where the derived class passes 
    // itself as a template argument to the base class(es). Here, it seemed
    // more convenient to pass up just the traversal tag of the most 
    // derived class.
    //
    template<
      class Value,
      class Traversal,
      class Reference,
      class Difference,
      class UsedAsBaseForTraversal = Traversal
    >
    class any_iterator_abstract_base;

    ///////////////////////////////////////////////////////////////////////
    // 
    template<
      class Value,
      class Reference,
      class Difference,
      class UsedAsBaseForTraversal
    >
    class any_iterator_abstract_base<
      Value,
      boost::incrementable_traversal_tag,
      Reference,
      Difference,
      UsedAsBaseForTraversal
    >
    {

    protected:
      typedef any_iterator_abstract_base<
        Value,
        UsedAsBaseForTraversal,
        Reference,
        Difference
      > most_derived_type;

      typedef most_derived_type clone_result_type;

      typedef any_iterator_abstract_base<
        typename boost::add_const<Value>::type,
        UsedAsBaseForTraversal,
        typename make_iterator_reference_const<Reference>::type,
        Difference
      > const_clone_with_const_value_type_result_type;

      typedef any_iterator_abstract_base<
        typename boost::remove_const<Value>::type,
        UsedAsBaseForTraversal,
        typename make_iterator_reference_const<Reference>::type,
        Difference
      > const_clone_with_non_const_value_type_result_type;

    public:

      // Plain clone function for copy construction and assignment.
      virtual clone_result_type * clone() const=0;
  
      // Clone functions for conversion to a const iterator
      virtual const_clone_with_const_value_type_result_type * make_const_clone_with_const_value_type() const=0;
      virtual const_clone_with_non_const_value_type_result_type * make_const_clone_with_non_const_value_type() const=0;

      // gcc 3.4.2 does not like pure virtual declaration with inline definition,
      // so I make the destructor non-pure just to spite them.
      virtual ~any_iterator_abstract_base()
      {}

      virtual Reference dereference() const=0;
      virtual void increment() = 0;

    };

    ///////////////////////////////////////////////////////////////////////
    // 
    template<
      class Value,
      class Reference,
      class Difference,
      class UsedAsBaseForTraversal
    >
    class any_iterator_abstract_base<
      Value,
      boost::single_pass_traversal_tag,
      Reference,
      Difference,
      UsedAsBaseForTraversal
    > : public any_iterator_abstract_base<
          Value,
          boost::incrementable_traversal_tag,
          Reference,
          Difference,
          UsedAsBaseForTraversal
        >
    {

    public:

      // gcc 3.4.2 insists on qualification of most_derived_type.
      virtual bool equal(typename any_iterator_abstract_base::most_derived_type const &) const = 0;

      virtual any_iterator_abstract_base<
        Value,
        boost::incrementable_traversal_tag,
        Reference,
        Difference
      >* make_incrementable_version() const=0;
    };

    ///////////////////////////////////////////////////////////////////////
    // 
    template<
      class Value,
      class Reference,
      class Difference,
      class UsedAsBaseForTraversal
    >
    class any_iterator_abstract_base<
      Value,
      boost::forward_traversal_tag,
      Reference,
      Difference,
      UsedAsBaseForTraversal
    > : public any_iterator_abstract_base<
          Value,
          boost::single_pass_traversal_tag,
          Reference,
          Difference,
          UsedAsBaseForTraversal
        >
    {
    public:
      virtual any_iterator_abstract_base<
        Value,
        boost::single_pass_traversal_tag,
        Reference,
        Difference
      >* make_single_pass_version() const=0;
    };

    ///////////////////////////////////////////////////////////////////////
    // 
    template<
      class Value,
      class Reference,
      class Difference,
      class UsedAsBaseForTraversal
    >
    class any_iterator_abstract_base<
      Value,
      boost::bidirectional_traversal_tag,
      Reference,
      Difference,
      UsedAsBaseForTraversal
    > : public any_iterator_abstract_base<
          Value,
          boost::forward_traversal_tag,
          Reference,
          Difference,
          UsedAsBaseForTraversal
        >
    {

    public:
      
      virtual void decrement() = 0;

      virtual any_iterator_abstract_base<
        Value,
        boost::forward_traversal_tag,
        Reference,
        Difference
      >* make_forward_version() const=0;
    };

    ///////////////////////////////////////////////////////////////////////
    // 
    template<
      class Value,
      class Reference,
      class Difference,
      class UsedAsBaseForTraversal
    >
    class any_iterator_abstract_base<
      Value,
      boost::random_access_traversal_tag,
      Reference,
      Difference,
      UsedAsBaseForTraversal
    > : public any_iterator_abstract_base<
          Value,
          boost::bidirectional_traversal_tag,
          Reference,
          Difference,
          UsedAsBaseForTraversal
        >
    {

    public:

      virtual void advance(Difference) = 0;

      // gcc 3.4.2 insists on qualification of most_derived_type.
      virtual Difference distance_to(typename any_iterator_abstract_base::most_derived_type const &) const= 0;

      virtual any_iterator_abstract_base<
        Value,
        boost::bidirectional_traversal_tag,
        Reference,
        Difference
      >* make_bidirectional_version() const=0;
    };

  } // end namespace detail

} // end namespace IteratorTypeErasure

#endif // ANY_ITERATOR_ABSTRACT_BASE_01102007TMB_HPP
